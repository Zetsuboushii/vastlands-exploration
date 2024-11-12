#!/usr/bin/bash

if [ "$(basename "$PWD")" != "vastlands-exploration" ]; then
  echo "You must be inside the root folder ('vastlands-exploration') for this script to work."
  exit 1
fi

if [ ! -e "infrastructure/.env" ]; then
  cp infrastructure/sample.env infrastructure/.env
  nano infrastructure/.env # vim users can be expected to not need this script :P
fi

export $(cat infrastructure/.env)

if [ ! -d "venv" ]; then
  echo "Setting up virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "Installing necessary packages..."
pip3 install -r requirements.txt

echo "Setting up database..."

docker compose -f infrastructure/docker-compose.yml up -d

read -p "Include map plots? Requires faergria map to be hosted locally. [Y/n] " response

case "${response,,}" in
    y|yes|"")
        suffix=""
        ;;
    n|no)
        suffix=" -s"
        ;;
    *)  # Catch-all for invalid input
        echo "Invalid input. Please enter 'y' or 'n'."
        exit 1
        ;;
esac

base_command="python3 main.py$suffix"

echo "Loading data into database..."

export $(cat infrastructure/.env)
eval "$base_command -f load"

echo "Vastlands Explorer is ready."
echo "You can now execute '$base_command' with the corresponding argument (plot or serve). See '$base_command --help' for more info."