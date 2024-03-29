# bot.py
#!/usr/bin/env python3

import os
import random
import discord
import datetime

from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()

bot = commands.Bot(command_prefix="$", intents=intents)

gif_shared = False
update_time = datetime.datetime.now()

# Dog Command
# Sends a picture of the dogs. 
@bot.command(name='dog', help='Uploads the most recent picture of the dogs')
async def dog_detect(ctx):

    filepath = "data/output_images/saved_dog_pic.jpg"
    response = "Here is the most recent picture of Maggie and Monty!"
    await ctx.send(response, file=discord.File(filepath))

# Cat Command
# Sends a picture of a cat. 
@bot.command(name='cat', help='Uploads the most recent picture of a cat')
async def dog_detect(ctx):

    filepath = "data/output_images/cat.jpg"
    response = "BARK BARK BARK BARK!"
    await ctx.send(response, file=discord.File(filepath))

# both dogs command
# Sends a picture that detected both dogs. 
@bot.command(name='both', help='Uploads the most recent picture that detected both dogs')
async def dog_detect(ctx):

    filepath = "data/output_images/both_dogs.jpg"
    response = "Best Friends!"
    await ctx.send(response, file=discord.File(filepath))


# GIF Command
# Send the most recent GIF of the the dogs regardless of if it has been sent before
@bot.command(name='gif', help='Uploads the most recent gif of the dogs')
async def dog_detect(ctx):

    global update_time
    filepath = "data/output_images/saved_dog.gif"
    
    m_time = os.path.getmtime(filepath)
    dt_m = datetime.datetime.fromtimestamp(m_time)
    update_time = dt_m

    response = "Here is a gif of Maggie and Monty! Last updated at " + str(dt_m)
    await ctx.send(response, file=discord.File(filepath))


# Update Command
# Sends the most recent GIF of the dogs if it has not been sent yet. 
# TODO: Have it send the latest gif right when it updates
@bot.command(name='update', help='Will look for the most recent GIF and share it. \
    If the GIF has already been shared via $GIF it will wait until a new one is made. ')
async def dog_detect(ctx):
    
    global update_time
    filepath = "data/output_images/saved_dog.gif"

    m_time = os.path.getmtime(filepath)
    dt_m = datetime.datetime.fromtimestamp(m_time)

    if update_time != dt_m:
        update_time = dt_m
        response = "Here is the newest GIF of Maggie and Monty!"
        await ctx.send(response, file=discord.File(filepath))
    else:
        response = "There is no new GIF of the Corgis."
        await ctx.send(response)


bot.run(TOKEN)