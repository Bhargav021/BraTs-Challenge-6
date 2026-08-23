# Optional: give Claude Code a real shell on the Colab VM

The Drive-mirror channel is enough for launching runs and reading their output, and it
is the one that keeps working when the browser tab is closed. Use SSH only when you
want Claude to run interactive commands on the GPU box — `nvidia-smi` mid-run, a quick
`python -c` shape check, poking at a crashed process.

Trade-offs, honestly: this depends on a tunnel that dies with the Colab session, free
tier sessions get preempted, and running long-lived tunnels is the kind of thing
Colab's terms have historically been unenthusiastic about. Check the current Colab
terms before relying on it, and never make it the only channel.

## In Colab (one cell)

```python
!pip install -q colab-ssh --upgrade
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="<pick-a-strong-one>")
```

It prints a hostname and port. Put them into your local `~/.ssh/config`:

```
Host colab
    HostName <printed-hostname>
    User root
    Port <printed-port>
    ProxyCommand cloudflared access ssh --hostname %h
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

## On your machine

Verify by hand first: `ssh colab nvidia-smi`

Then allow it in `.claude/settings.json` so Claude does not need a prompt each time:

```json
{ "permissions": { "allow": ["Bash(ssh colab:*)"] } }
```

Tell Claude in your session (or add to `configs/paths.yaml`) that `ssh colab` is live.
The `colab-runner` agent will use it for one-off inspection commands, and still use
git + the Drive mirror for launching and recording runs.

When the session dies, the host is gone. Re-run the Colab cell and update the port.
