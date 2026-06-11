# Phase 10 Notes

Phase 10 adds a controlled tool/action layer on top of goals and plans.

## Intent

Plans in phase 9 could identify the next unblocked step, but they could not execute anything.
Phase 10 introduces a narrow action surface that stays safe and inspectable:

- dry-run by default
- explicit commit mode
- persistent execution records
- step-to-tool resolution for safe internal actions

## Current built-in tools

- diagnostics
- search_memory
- remember
- sense
- watch
- consolidate
- reflect
- sleep
- dream
- speak
- note
- goal_add / goal_focus
- plan_add / plan_focus
- step_done / step_doing / step_block

## Non-goals

This is not an unrestricted shell, file-write, or network automation layer.
The purpose here is to give WonderBot a reliable execution contract before broadening the tool surface.
