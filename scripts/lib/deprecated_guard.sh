#!/usr/bin/env bash

cppmega_deprecated_script_guard() {
  local script_name="$1"
  local replacement="$2"
  local ack_env="${3:-CPPMEGA_I_UNDERSTAND_LEGACY_H200_SCRIPTS_ARE_DEPRECATED}"

  if [[ "${!ack_env:-0}" != "1" ]]; then
    {
      echo "FATAL: ${script_name} is DEPRECATED and disabled by default."
      echo "Use: ${replacement}"
      echo "To force this old script, set ${ack_env}=1."
    } >&2
    exit 2
  fi

  echo "DEPRECATED: ${script_name} enabled because ${ack_env}=1. Use ${replacement}." >&2
}
