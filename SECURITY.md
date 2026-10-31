# Security and patient data

MentoPlan runs locally and binds to `127.0.0.1` by default. It has no authentication: do not expose
the server to a network that untrusted people can reach.

Patient scans and derived data stay in `data/` on your machine and are excluded from git.
Never attach identifiable scans or images to issues or pull requests.

To report a vulnerability, open a private
[security advisory](https://github.com/debarch777/mentoplan/security/advisories/new).
