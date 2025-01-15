#!/bin/bash

echo kill from $1 to $2...
# Loop through the range and send SIGTERM to each process
for ((pid = $1; pid <= $2; pid++)); do
    kill -9 $pid
done
