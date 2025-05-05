# Overview

Author: Torda Balázs  
Title: CanonFodder


## What is this?

This is a code base accompanying a written assignment for IU.<br>
For further elaboration, you can contact balazs.torda@iu-study.org.

## The basic building blocks

It is a project developed in PyCharm incorporating Python scripts and SQL-based db connections.<br>
As an example for fetching scrobble data in CSV, I am using https://benjaminbenben.com/lastfm-to-csv/.<br>
You can look at Ben Foxall's code at https://github.com/benfoxall/lastfm-to-csv.<br>
For correct data retrieval, I am using the last.fm API.<br>
The terms of use can be checked at https://www.last.fm/api.<br>

## Installation

Note: Make sure you have Python 3.12 installed on your system.<br>
Clone the repository.<br>
```shell
docker-compose up --build
```
Create a virtual environment manually using a command like:<br>
```shell
python -m venv .venv
```
Activate the virtual environment:<br>
```shell
.venv\Scripts\activate
```
or
```shell
source .venv/bin/activate
```
Navigate to the project directory, then launch<br>
```shell
pip install -r requirements.txt
```
or
```shell
pip install .
```
Copy `.env.example` ➜ `.env` and fill in the required values.<br> 
      * Get a free key at https://www.last.fm/api/account/create<br> 
      * For read-only demos only `LASTFM_API_KEY` is mandatory.<br>
Alembic and SQLAlchemy is used to allow for multiple DB backends.<br>
I am using MySQL with setting DB_URL=mysql+pymysql://user:pass@localhost/canonfodder,
but if you look in DB/common.py, you will see that the bootstrapping is configured with a local fallback to sqlite.<br>
DB_URL there is set to sqlite:///canonfodder.db automatically if no MySQL is given in .env.

## How to use it?
Command
```shell
python main.py
```
for a complete data fetch.

I am committing example .parquets for fast fload, so if you are in a hurry,
you can directly command
```shell
python dev_profile.py
```
&
```shell
python dev_canon.py
```
and check out in a notebook-style step-by-step way how is canonization envisaged in this project.
