# Add this to ~/.bashrc or ~/.zshrc.
# It creates one stable session id per terminal window.

if [ -z "$DOIT_SESSION_ID" ]; then
  export DOIT_SESSION_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
fi

# Optional secure-mode override:
# export DOIT_SECURE=true
