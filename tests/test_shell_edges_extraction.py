"""Tests for _shell_edges_from_script regex-based shell edge extraction."""

from __future__ import annotations

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
$Root = "./src"
Get-ChildItem -Path $Root -Filter *.cpp | Where-Object { $_.Length -gt 0 }
Copy-Item -Source ./a.txt -Destination ./b.txt
Write-Output $root > ./result.txt
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
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in pipe_edges
        } >= {
            (BASH_FIXTURE.index("cat"), BASH_FIXTURE.index("grep")),
            (BASH_FIXTURE.index("grep"), BASH_FIXTURE.index("sort")),
        }

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
        definition = BASH_FIXTURE.index("BUILD_DIR")
        assert any(
            edge["from_char"] == definition
            and BASH_FIXTURE[edge["to_char"] :].startswith("$BUILD_DIR")
            for edge in var_edges
        )

    def test_redirect_edges_detected(self) -> None:
        text = (
            'echo "quoted > text" # comment > ignored.txt\n'
            "cat < input.txt; sort > output.txt\n"
            "cat input.txt | sort > piped.txt\n"
            'printf ok > "quoted result.txt"\n'
        )
        edges = ip._shell_edges_from_script(text, "bash")
        redirects = {
            (edge["from_char"], edge["to_char"], edge["kind"])
            for edge in edges
            if edge["kind"]
            in {
                int(DomainEdgeKind.SHELL_REDIR_IN),
                int(DomainEdgeKind.SHELL_REDIR_OUT),
            }
        }
        first_cat = text.index("cat")
        first_sort = text.index("sort")
        second_sort = text.index("sort", first_sort + 1)
        printf = text.index("printf")
        assert redirects == {
            (
                first_cat,
                text.index("input.txt", first_cat),
                int(DomainEdgeKind.SHELL_REDIR_IN),
            ),
            (
                first_sort,
                text.index("output.txt"),
                int(DomainEdgeKind.SHELL_REDIR_OUT),
            ),
            (
                second_sort,
                text.index("piped.txt"),
                int(DomainEdgeKind.SHELL_REDIR_OUT),
            ),
            (
                printf,
                text.index('"quoted result.txt"'),
                int(DomainEdgeKind.SHELL_REDIR_OUT),
            ),
        }

    def test_control_punctuation_starts_comment_for_pipe_and_redirect_scan(self) -> None:
        text = "echo ok;# ignored > fake.txt | nope\n"
        edges = ip._shell_edges_from_script(text, "bash")

        assert all(
            edge["kind"]
            not in {
                int(DomainEdgeKind.SHELL_PIPE),
                int(DomainEdgeKind.SHELL_REDIR_OUT),
            }
            for edge in edges
        )

    def test_pipeline_anchors_respect_statement_boundaries_and_line_continuation(
        self,
    ) -> None:
        posix = "echo prep; cat input.txt | sort\n"
        posix_edges = ip._shell_edges_from_script(posix, "bash")
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in posix_edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        } == {(posix.index("cat"), posix.index("sort"))}

        powershell = "Get-ChildItem |\n  Where-Object { $_.Length -gt 0 }\n"
        powershell_edges = ip._shell_edges_from_script(
            powershell,
            "powershell",
        )
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in powershell_edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        } == {
            (
                powershell.index("Get-ChildItem"),
                powershell.index("Where-Object"),
            )
        }

    def test_control_flow_and_brace_expansion_keep_inner_command_anchors(
        self,
    ) -> None:
        bash = (
            "if true; then cat input.txt | sort; fi\n"
            "echo {a,b} | sort\n"
            'case "$kind" in a) cat input.txt | sort;; esac\n'
        )
        bash_pipes = {
            (edge["from_char"], edge["to_char"])
            for edge in ip._shell_edges_from_script(bash, "bash")
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        }
        assert bash_pipes == {
            (bash.index("cat"), bash.index("sort")),
            (bash.index("echo"), bash.index("sort", bash.index("echo"))),
            (bash.rindex("cat"), bash.rindex("sort")),
        }

        powershell = (
            "if ($ok) { Get-ChildItem | Sort-Object }\n"
            "ForEach-Object { Get-Item . | Sort-Object }\n"
            "if ($ok) { Write-Output hi > out.txt }\n"
        )
        powershell_edges = ip._shell_edges_from_script(
            powershell,
            "powershell",
        )
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in powershell_edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        } == {
            (
                powershell.index("Get-ChildItem"),
                powershell.index("Sort-Object"),
            ),
            (
                powershell.index("Get-Item"),
                powershell.rindex("Sort-Object"),
            ),
        }
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in powershell_edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_REDIR_OUT)
        } == {
            (
                powershell.index("Write-Output"),
                powershell.index("out.txt"),
            )
        }

    def test_shell_variable_edges_ignore_non_interpolating_regions(self) -> None:
        for shell_kind, definition in (
            ("bash", "VALUE=1"),
            ("tcsh", "setenv VALUE 1"),
        ):
            text = (
                f"{definition}\n"
                "echo '$VALUE'\n"
                "# $VALUE\n"
                'echo "$VALUE"\n'
            )
            variable_edges = [
                edge
                for edge in ip._shell_edges_from_script(text, shell_kind)
                if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
            ]
            assert variable_edges == [
                {
                    "from_char": text.index("VALUE"),
                    "to_char": text.rindex("$VALUE"),
                    "kind": int(DomainEdgeKind.SHELL_VAR_DEF_USE),
                }
            ]

    def test_posix_heredoc_masks_structure_and_respects_interpolation(self) -> None:
        text = (
            "VALUE=1\n"
            "BITS=2\n"
            "cat <<'LITERAL'\n"
            "fake | pipe\n"
            "source ./fake.sh\n"
            "$VALUE\n"
            "LITERAL\n"
            "cat <<EXPANDED\n"
            "$VALUE\n"
            r"\$VALUE"
            "\nEXPANDED\n"
            "cat <<<EOF\n"
            "echo $((1 << BITS))\n"
            "echo real | sort\n"
        )
        edges = ip._shell_edges_from_script(text, "bash")

        assert {
            (edge["from_char"], edge["to_char"])
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        } == {(text.index("echo real"), text.rindex("sort"))}
        assert all(
            not (
                edge["kind"] == int(DomainEdgeKind.SHELL_COMMAND_FILE)
                and edge["to_char"] == text.index("./fake.sh")
            )
            for edge in edges
        )
        assert {
            edge["to_char"]
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        } == {text.index("$VALUE", text.index("EXPANDED"))}

    def test_multiline_literals_and_comments_do_not_emit_shell_edges(self) -> None:
        powershell = (
            "<#\n"
            "Get-ChildItem -Path ./fake.txt | Out-File ./fake.out\n"
            "#>\n"
            '@"\n'
            "literal | fake\n"
            '"@\n'
            "Write-Output live | Out-File ./real.txt\n"
        )
        powershell_edges = ip._shell_edges_from_script(
            powershell,
            "powershell",
        )
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in powershell_edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        } == {
            (
                powershell.index("Write-Output live"),
                powershell.index("Out-File ./real"),
            )
        }
        fake_spans = (
            range(
                powershell.index("Get-ChildItem"),
                powershell.index("#>"),
            ),
            range(
                powershell.index('@"'),
                powershell.index('"@') + 2,
            ),
        )
        assert all(
            not any(
                edge[endpoint] in span
                for span in fake_spans
                for endpoint in ("from_char", "to_char")
            )
            for edge in powershell_edges
        )

        posix = "'quoted\nsource ./fake.sh\n'\nsource ./real.sh\n"
        source_edges = [
            edge
            for edge in ip._shell_edges_from_script(posix, "bash")
            if edge["kind"] == int(DomainEdgeKind.SHELL_COMMAND_FILE)
        ]
        assert source_edges == [
            {
                "from_char": posix.rindex("source"),
                "to_char": posix.index("./real.sh"),
                "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE),
            }
        ]

    def test_powershell_comment_markers_require_token_boundaries(self) -> None:
        text = (
            "Write-Output Foo#Bar | Sort-Object\n"
            "Write-Output Foo<#Bar#> | Sort-Object\n"
            "$Value=1\n"
            "$Other=<# ignored $Value | fake #>2\n"
            "Write-Output $value\n"
        )
        edges = ip._shell_edges_from_script(text, "powershell")

        assert {
            (edge["from_char"], edge["to_char"])
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        } == {
            (text.index("Write-Output"), text.index("Sort-Object")),
            (
                text.index("Write-Output", 1),
                text.index("Sort-Object", text.index("Sort-Object") + 1),
            ),
        }
        assert {
            edge["to_char"]
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        } == {text.rindex("$value")}

    def test_powershell_here_string_escaped_variable_is_literal(self) -> None:
        text = (
            "$Value=1\n"
            '$Text=@"\n'
            "live=$VALUE escaped=`$Value\n"
            '"@\n'
        )
        variable_edges = [
            edge
            for edge in ip._shell_edges_from_script(text, "powershell")
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]

        assert variable_edges == [
            {
                "from_char": text.index("$Value"),
                "to_char": text.index("$VALUE"),
                "kind": int(DomainEdgeKind.SHELL_VAR_DEF_USE),
            }
        ]

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

    def test_powershell_dialect_edges_keep_command_and_variable_anchors(self) -> None:
        edges = ip._shell_edges_from_script(POWERSHELL_FIXTURE, "powershell")
        pipeline = [
            edge
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
        ]
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in pipeline
        } >= {
            (
                POWERSHELL_FIXTURE.index("Get-ChildItem"),
                POWERSHELL_FIXTURE.index("Where-Object"),
            )
        }

        variable_edges = [
            edge
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]
        definition = POWERSHELL_FIXTURE.index("$Root")
        assert any(
            edge["from_char"] == definition
            and POWERSHELL_FIXTURE[edge["to_char"] :].lower().startswith("$root")
            for edge in variable_edges
        )

    def test_powershell_assignment_is_not_misreported_as_a_variable_use(self) -> None:
        text = "$Value = 1\nWrite-Output $value\n$VALUE = 2\nWrite-Output $Value\n"
        edges = ip._shell_edges_from_script(text, "powershell")
        variable_edges = [
            edge
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]

        first_definition = text.index("$Value")
        second_definition = text.index("$VALUE")
        first_use = text.index("$value")
        second_use = text.rindex("$Value")
        assert {
            (edge["from_char"], edge["to_char"])
            for edge in variable_edges
        } == {
            (first_definition, first_use),
            (second_definition, second_use),
        }

    def test_powershell_typed_and_parameter_definitions_link_to_uses(self) -> None:
        text = (
            "param(\n"
            "  [Parameter(Mandatory=$true)]\n"
            '  [ValidateSet("fast", "safe")]\n'
            "  [string]$Mode\n"
            ")\n"
            '[string]$Root = "./src"\n'
            "Write-Output $mode\n"
            "Write-Output $root\n"
        )
        variable_edges = [
            edge
            for edge in ip._shell_edges_from_script(text, "powershell")
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]

        assert {
            (edge["from_char"], edge["to_char"])
            for edge in variable_edges
        } == {
            (text.index("$Mode"), text.index("$mode")),
            (text.index("$Root"), text.index("$root")),
        }

    def test_powershell_variable_edges_ignore_comments_and_single_quotes(self) -> None:
        text = (
            "$Value = 1\n"
            "Write-Output '$value'\n"
            "# $VALUE\n"
            'Write-Output "live=$vAlUe"\n'
        )
        edges = ip._shell_edges_from_script(text, "powershell")
        variable_edges = [
            edge
            for edge in edges
            if edge["kind"] == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
        ]

        assert variable_edges == [
            {
                "from_char": text.index("$Value"),
                "to_char": text.rindex("$vAlUe"),
                "kind": int(DomainEdgeKind.SHELL_VAR_DEF_USE),
            }
        ]

    def test_powershell_fallback_handles_braced_scoped_and_compound_variables(
        self,
    ) -> None:
        from cppmega.data.build_parsers.base import ParsedDomainDocument
        from cppmega.data.build_parsers.shell import (
            _heuristic_assign_powershell_roles_and_edges,
        )
        from cppmega.data.domain_schema import DomainKind, DomainRoleKind

        text = (
            "param(\n"
            "  [Parameter(Mandatory=$true)]\n"
            '  [ValidateSet("fast", "safe")]\n'
            "  [string]$Mode\n"
            ")\n"
            '[string]$Typed = "./typed"\n'
            '${Root} = "./src"\n'
            "$env:Path += ';C:\\tools'\n"
            "$Items = Get-ChildItem -Path ${Root} | "
            "Where-Object { $_.Length -gt 0 }\n"
            "Write-Output $ENV:PATH\n"
            "Write-Output $mode\n"
            "Write-Output $typed\n"
            "Write-Output previous\n"
            "$Items | Select-Object Name\n"
            'Write-Output ok > "./result file.txt"\n'
            'Write-Error bad 2>> "err.log"\n'
        )
        parsed = ParsedDomainDocument.new(domain=DomainKind.SH, text=text)
        _heuristic_assign_powershell_roles_and_edges(parsed)
        edge_tokens = [
            (parsed.tokens[source].text, parsed.tokens[target].text, kind)
            for source, target, kind in parsed.edges
        ]

        assert sum(
            kind == int(DomainEdgeKind.SHELL_VAR_DEF_USE)
            for _source, _target, kind in edge_tokens
        ) == 5
        assert (
            "Get-ChildItem",
            "Where-Object",
            int(DomainEdgeKind.SHELL_PIPE),
        ) in edge_tokens
        assert (
            "$Items",
            "Select-Object",
            int(DomainEdgeKind.SHELL_PIPE),
        ) in edge_tokens
        assert (
            "$Mode",
            "$mode",
            int(DomainEdgeKind.SHELL_VAR_DEF_USE),
        ) in edge_tokens
        assert (
            "$Typed",
            "$typed",
            int(DomainEdgeKind.SHELL_VAR_DEF_USE),
        ) in edge_tokens
        assert (
            "Write-Output",
            "Select-Object",
            int(DomainEdgeKind.SHELL_PIPE),
        ) not in edge_tokens
        assert (
            "Write-Output",
            '"./result file.txt"',
            int(DomainEdgeKind.SHELL_REDIR_OUT),
        ) in edge_tokens
        assert edge_tokens.count(
            (
                "Write-Error",
                '"err.log"',
                int(DomainEdgeKind.SHELL_REDIR_OUT),
            )
        ) == 1
        assert all(
            kind != int(DomainEdgeKind.SHELL_COMMAND_FILE)
            for _source, _target, kind in edge_tokens
        )
        false_command_tokens = {"{", "}", ":", "+", "=", ".Length", "Root"}
        assert all(
            parsed.role_ids[index] != int(DomainRoleKind.COMMAND)
            for index, token in enumerate(parsed.tokens)
            if token.text in false_command_tokens
        )

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

    def test_canonical_shell_parser_and_merge_ignore_non_code_regions(self) -> None:
        from cppmega.data.domain_ingestion import parse_domain_document

        text = (
            "# fake | pipe\n"
            "'\n"
            "source ./quoted-fake.sh\n"
            "'\n"
            "cat <<'LITERAL'\n"
            "heredoc-fake | upload\n"
            "source ./heredoc-fake.sh\n"
            "if unbalanced\n"
            "LITERAL\n"
            "source ./real.sh\n"
            "echo real | sort\n"
        )
        fake_start = text.index("# fake")
        real_start = text.index("source ./real.sh")

        parsed = parse_domain_document("scripts/run.sh", text)
        parsed_edges = [
            {
                "from_char": parsed.tokens[source].start,
                "to_char": parsed.tokens[target].start,
                "kind": kind,
            }
            for source, target, kind in parsed.edges
        ]
        doc = ip.build_build_doc(
            "scripts/run.sh",
            text,
            "bash",
            project_id="test-owner/test-repo",
        )

        for edges in (parsed_edges, doc["shell_edges"]):
            assert all(
                edge["from_char"] < fake_start
                or edge["from_char"] >= real_start
                for edge in edges
            )
            assert all(
                edge["to_char"] < fake_start
                or edge["to_char"] >= real_start
                for edge in edges
            )
            assert {
                (edge["from_char"], edge["to_char"])
                for edge in edges
                if edge["kind"] == int(DomainEdgeKind.SHELL_PIPE)
            } == {(text.index("echo real"), text.index("sort"))}
            assert {
                edge["to_char"]
                for edge in edges
                if edge["kind"] == int(DomainEdgeKind.SHELL_COMMAND_FILE)
            } == {text.index("./real.sh")}

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

    def test_build_build_doc_powershell_uses_shared_shell_domain_with_dialect(self) -> None:
        from cppmega.data.domain_schema import DomainKind

        doc = ip.build_build_doc(
            "scripts/run.ps1",
            POWERSHELL_FIXTURE,
            "powershell",
            project_id="test-owner/test-repo",
        )

        assert doc["doc_type"] == "shell"
        assert doc["domain_kind"] == int(DomainKind.SH)
        assert doc["build_kind"] == "powershell"
        assert doc["language_info"]["primary_language"] == "powershell"
        assert doc["language_info"]["primary_dialect"] == "powershell"
        assert doc["domain_parse_info"]["parser_adapter"] == "powershell"
        assert doc["domain_parse_info"]["shell_dialect"] == "powershell"
        assert doc["domain_parse_info"]["shared_domain"] == "sh"
        assert doc["domain_parse_info"]["parse_engine"] != "posix-sh"
        assert doc["shell_edges"]

    def test_find_shell_files_covers_powershell_extensions_and_shebang(
        self,
        tmp_path,
    ) -> None:
        fixtures = {
            "build.ps1": "Write-Output ok\n",
            "module.psm1": "function Invoke-Build { }\n",
            "manifest.psd1": "@{ RootModule = 'module.psm1' }\n",
            "run": "#!/usr/bin/env pwsh\nWrite-Output ok\n",
            "build.bat": "@echo off\necho ok\n",
            "build.cmd": "@echo off\necho ok\n",
        }
        for name, text in fixtures.items():
            (tmp_path / name).write_text(text, encoding="utf-8")

        assert {
            (path.rsplit("/", 1)[-1], dialect)
            for path, dialect in ip.find_shell_files(str(tmp_path))
        } == {
            (
                name,
                "cmd" if name.endswith((".bat", ".cmd")) else "powershell",
            )
            for name in fixtures
        }

    def test_powershell_control_flow_is_not_validated_as_posix_shell(self) -> None:
        from cppmega.data.domain_ingestion import parse_domain_document
        from cppmega.data.domain_schema import DomainKind, ParseConfidence

        parsed = parse_domain_document(
            "scripts/conditional.ps1",
            "if ($true) { Write-Output 'ok' }\n",
        )

        assert parsed.domain == DomainKind.SH
        assert parsed.metadata["parser_adapter"] == "powershell"
        assert parsed.metadata["shell_dialect"] == "powershell"
        assert "malformed_powershell_shell" not in parsed.metadata.get(
            "raw_reason",
            "",
        )
        assert set(parsed.confidence_ids) != {int(ParseConfidence.RAW)}

    def test_powershell_utf8_byte_edges_attach_to_character_tokens(self) -> None:
        from cppmega.data.build_parsers.base import ParsedDomainDocument
        from cppmega.data.build_parsers.shell import _attach_edges_to_doc
        from cppmega.data.domain_ingestion import parse_domain_document
        from cppmega.data.domain_schema import DomainKind

        text = (
            "# Привет, мир\n"
            '$Root = "./src"\n'
            "Write-Output $root | Out-File ./result.txt\n"
        )
        source = text.encode("utf-8")
        direct = ParsedDomainDocument.new(domain=DomainKind.SH, text=text)
        _attach_edges_to_doc(
            direct,
            [
                (
                    source.index(b"$Root"),
                    source.index(b"$root"),
                    int(DomainEdgeKind.SHELL_VAR_DEF_USE),
                )
            ],
            source,
        )
        assert [
            (direct.tokens[source_idx].text, direct.tokens[target_idx].text, kind)
            for source_idx, target_idx, kind in direct.edges
        ] == [
            (
                "$Root",
                "$root",
                int(DomainEdgeKind.SHELL_VAR_DEF_USE),
            )
        ]

        # Exercise the real grammar when the optional parser pack is installed;
        # the direct attachment assertion above remains the dependency-free gate.
        parsed = parse_domain_document("scripts/unicode.ps1", text)
        edge_tokens = {
            (parsed.tokens[source].text, parsed.tokens[target].text, kind)
            for source, target, kind in parsed.edges
        }

        if parsed.metadata["parse_engine"] == "tree-sitter":
            assert (
                "$Root",
                "$root",
                int(DomainEdgeKind.SHELL_VAR_DEF_USE),
            ) in edge_tokens
            assert (
                "Write-Output",
                "Out-File",
                int(DomainEdgeKind.SHELL_PIPE),
            ) in edge_tokens
