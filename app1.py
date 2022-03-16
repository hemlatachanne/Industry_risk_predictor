#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:06:49 2022

@author: hemlata
"""

from flask import Flask,render_template,request

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")
@app.route("/index",methods=['POST'])
def submit():
    if request.method=="POST":
        desc = request.form["description"]
        return render_template("index.html",descrip = desc)


if __name__=="__main__":
    app.run(debug = True)