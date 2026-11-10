# Security

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on
the Security tab of this repository, not in a public issue. We aim to reply within five
working days.

Rung reads model API keys from the environment (`GEMINI_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`) and sends them only as request headers to the provider. Traces record
request bodies, never headers. If you find a key in a trace, report it as above.
