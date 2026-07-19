"""Tests for _shell_edges_from_script regex-based shell edge extraction."""

from __future__ import annotations

import pytest

from cppmega.data.domain_schema import DomainEdgeKind
from tools.clang_indexer import index_project as ip


BASH_FIXTURE = """\
#!/bin/bash
export BUILD_DIR=/tmp/build
source ./common.sh
. ./helpers.sh

gcc -o main main.c
cat input.txt | grep pattern | sort
echo $BUILD_DIR
make -C $BUILD_DIR all
"""

TCSH_FIXTURE = """\
#!/bin/tcsh
setenv MY_PATH /usr/local/bin
echo $MY_PATH
ls | grep foo
"""

POWERSHELL_FIXTURE = """\
Get-ChildItem -Path ./src -Filter *.cpp
Copy-Item -Source ./a.txt -Destination ./b.txt
"""


class TestShellEdgesFromScript:
    def test_bash_produces_nonempty_edges(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        assert len(edges) > 0

    def test_edge_structure_is_char_triple(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        for edge in edges:
            assert set(edge.keys()) == {"from_char", "to_char", "kind"}
            assert isinstance(edge["from_char"], int)
            assert isinstance(edge["to_char"], int)
            assert isinstance(edge["kind"], int)
            assert 0 <= edge["from_char"] < len(BASH_FIXTURE)
            assert 0 <= edge["to_char"] < len(BASH_FIXTURE)
            assert edge["from_char"] != edge["to_char"]

    def test_pipe_edges_detected(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        pipe_edges = [
            e for e in edges if e["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        ]
        # "cat input.txt | grep pattern | sort" has at least 2 pipe edges
        assert len(pipe_edges) >= 2

    def test_source_edges_detected(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        source_edges = [
            e
            for e in edges
            if e["kind"] == int(DomainEdgeKind.SHELL_COMMAND_FILE)
        ]
        # "source ./common.sh" and ". ./helpers.sh" produce source edges
        assert len(source_edges) >= 2

    def test_var_def_use_edges_detected(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        var_edges = [
            e
            for e in edges
            if e["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]
        # BUILD_DIR is defined and used via $BUILD_DIR
        assert len(var_edges) >= 1

    def test_command_file_edges_detected(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        cmd_file_edges = [
            e
            for e in edges
            if e["kind"] == int(DomainEdgeKind.SHELL_COMMAND_FILE)
        ]
        # "gcc -o main main.c" should produce a command→file edge
        assert len(cmd_file_edges) >= 1

    def test_tcsh_var_def_use(self) -> None:
        edges = ip._shell_edges_from_script(TCSH_FIXTURE, "tcsh")
        var_edges = [
            e
            for e in edges
            if e["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]
        assert len(var_edges) >= 1

    def test_tcsh_pipe_edges(self) -> None:
        edges = ip._shell_edges_from_script(TCSH_FIXTURE, "tcsh")
        pipe_edges = [
            e for e in edges if e["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        ]
        assert len(pipe_edges) >= 1

    def test_powershell_cmdlet_parameter_edges(self) -> None:
        edges = ip._shell_edges_from_script(POWERSHELL_FIXTURE, "powershell")
        cmd_edges = [
            e
            for e in edges
            if e["kind"] == int(DomainEdgeKind.SHELL_COMMAND_FILE)
        ]
        # Get-ChildItem -Path ./src and Copy-Item -Source/-Destination
        assert len(cmd_edges) >= 2

    def test_empty_text_produces_no_edges(self) -> None:
        edges = ip._shell_edges_from_script("", "bash")
        assert edges == []

    def test_no_duplicates(self) -> None:
        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        triples = [(e["from_char"], e["to_char"], e["kind"]) for e in edges]
        assert len(triples) == len(set(triples))

    def test_all_edge_kinds_are_valid_shell_family(self) -> None:
        from cppmega.data.domain_schema import domain_edge_family

        edges = ip._shell_edges_from_script(BASH_FIXTURE, "bash")
        for edge in edges:
            family = domain_edge_family(edge["kind"])
            assert family == "shell", (
                f"edge kind {edge['kind']} belongs to family {family!r}, expected 'shell'"
            )


class TestBuildBuildDocShellEdges:
    """Integration: build_build_doc populates shell_edges for shell docs."""

    def test_build_build_doc_shell_edges_populated(self) -> None:
        text = "#!/bin/bash\ncat foo.txt | sort\necho done\n"
        doc = ip.build_build_doc(
            "scripts/run.sh",
            text,
            "bash",
            project_id="test-owner/test-repo",
        )
        shell_edges = doc.get("shell_edges", [])
        assert len(shell_edges) > 0
        for edge in shell_edges:
            assert "from_char" in edge
            assert "to_char" in edge
            assert "kind" in edge

    def test_build_build_doc_non_shell_has_empty_shell_edges(self) -> None:
        text = "cmake_minimum_required(VERSION 3.16)\nproject(foo)\n"
        doc = ip.build_build_doc(
            "CMakeLists.txt",
            text,
            "cmake",
            project_id="test-owner/test-repo",
        )
        shell_edges = doc.get("shell_edges", [])
        assert shell_edges == []
