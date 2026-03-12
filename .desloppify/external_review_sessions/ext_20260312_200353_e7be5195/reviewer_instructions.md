# External Blind Review Session

Session id: ext_20260312_200353_e7be5195
Session token: 2ab81654cd4cf56c72aad5112b167a9f
Blind packet: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/review_packet_blind.json
Template output: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_200353_e7be5195/review_result.template.json
Claude launch prompt: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_200353_e7be5195/claude_launch_prompt.md
Expected reviewer output: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_200353_e7be5195/review_result.json

Happy path:
1. Open the Claude launch prompt file and paste it into a context-isolated subagent task.
2. Reviewer writes JSON output to the expected reviewer output path.
3. Submit with the printed --external-submit command.

Reviewer output requirements:
1. Return JSON with top-level keys: session, assessments, issues.
2. session.id must be `ext_20260312_200353_e7be5195`.
3. session.token must be `2ab81654cd4cf56c72aad5112b167a9f`.
4. Include issues with required schema fields (dimension/identifier/summary/related_files/evidence/suggestion/confidence).
5. Use the blind packet only (no score targets or prior context).
