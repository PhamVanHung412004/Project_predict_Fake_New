#!/bin/bash

# Kích hoạt virtual environment
source env_fake_new/bin/activate

# Cài đặt dependencies nếu chưa có
pip install -r backend/requirements.txt

# Chạy backend
cd backend
python app.py
