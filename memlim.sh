#!/bin/sh
ulimit -m 2048000
ulimit -v 2048000
ulimit -H -v
./a.out
