#!/bin/bash
/etc/init.d/postgresql start
psql -U postgres -d postgres -a -f init.sql
#exit
