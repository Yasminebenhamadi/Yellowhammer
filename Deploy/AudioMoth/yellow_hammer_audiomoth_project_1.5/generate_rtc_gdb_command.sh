#!/bin/bash

# Fetch the current UNIX timestamp
current_time=$(date +%s)

# Generate GDB command file
cat <<EOF > set_timekeeper.gdb
set timekeeper.unix_timestamp = $current_time
EOF

echo "GDB command file 'set_timekeeper.gdb' generated with timestamp: $current_time"