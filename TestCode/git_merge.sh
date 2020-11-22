#!/bin/bash

git fetch origin main
git reset --hard FETCH_HEAD
git clean -df