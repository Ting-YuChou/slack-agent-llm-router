# Run inside a ClickHouse container with `sh -c "$(cat this-file)" sh ARGS...`.
# clickhouse-client reads the CLICKHOUSE_USER and CLICKHOUSE_PASSWORD that
# compose gave the container from a private temporary config file, so the
# password is on no command line, on the host or in the container.
umask 077
d=$(mktemp -d) || exit 1
trap 'rm -rf "$d"' EXIT
trap 'exit 1' HUP INT TERM
q() { printf %s "$1" | sed "s/'/''/g"; }
printf "user: '%s'\npassword: '%s'\n" \
  "$(q "$CLICKHOUSE_USER")" "$(q "$CLICKHOUSE_PASSWORD")" > "$d/client.yaml"
clickhouse-client --config-file "$d/client.yaml" "$@"
