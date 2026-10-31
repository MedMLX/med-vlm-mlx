#!/bin/sh
# Shared path boundary for Makefile cleanup and source scans. Source this file.

validate_repo_paths() {
  repo_path_kind=$1
  shift
  for repo_path_entry do
    case "$repo_path_entry" in
      ""|/*)
        printf 'Unsafe %s entry: %s\n' "$repo_path_kind" "$repo_path_entry"
        return 1 ;;
    esac
    repo_path_prefix=.
    repo_path_remaining=$repo_path_entry
    while [ -n "$repo_path_remaining" ]; do
      repo_path_component=${repo_path_remaining%%/*}
      case "$repo_path_remaining" in
        */*) repo_path_remaining=${repo_path_remaining#*/} ;;
        *) repo_path_remaining= ;;
      esac
      case "$repo_path_component" in
        ""|.) continue ;;
        ..)
          printf 'Unsafe %s entry: %s\n' "$repo_path_kind" "$repo_path_entry"
          return 1 ;;
      esac
      repo_path_prefix=$repo_path_prefix/$repo_path_component
      if [ -L "$repo_path_prefix" ]; then
        printf 'Unsafe %s entry: %s (symbolic link)\n' "$repo_path_kind" "$repo_path_entry"
        return 1
      fi
    done
    if [ "$repo_path_kind" = GENERATED_PATHS ] && [ "$repo_path_prefix" = . ]; then
      printf 'Unsafe %s entry: %s (repository root)\n' "$repo_path_kind" "$repo_path_entry"
      return 1
    fi
  done
}
