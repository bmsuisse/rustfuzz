# External Blind Review Session

Session id: ext_20260312_203918_cbcc343a
Session token: 306dfa449ce832d20f627f89463bb2e3
Blind packet: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/review_packet_blind.json
Template output: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_203918_cbcc343a/review_result.template.json
Claude launch prompt: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_203918_cbcc343a/claude_launch_prompt.md
Expected reviewer output: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_203918_cbcc343a/review_result.json

Happy path:
1. Open the Claude launch prompt file and paste it into a context-isolated subagent task.
2. Reviewer writes JSON output to the expected reviewer output path.
3. Submit with the printed --external-submit command.

Reviewer output requirements:
1. Return JSON with top-level keys: session, assessments, issues.
2. session.id must be `ext_20260312_203918_cbcc343a`.
3. session.token must be `306dfa449ce832d20f627f89463bb2e3`.
4. Include issues with required schema fields (dimension/identifier/summary/related_files/evidence/suggestion/confidence).
5. Use the blind packet only (no score targets or prior context).
