#!/bin/sh
cd backend
gnome-terminal -e "python run0.py"
cd ../frontend
npm start
