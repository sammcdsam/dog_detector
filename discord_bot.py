
# bot.py
import os
import random
import discord

from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()

bot = commands.Bot(command_prefix="$", intents=intents)

@bot.command(name='dog', help='Uploads the most recent picture of the dogs')
async def dog_detect(ctx):

    filepath = "data/output_images/saved_dog_pic.jpg"
    response = "Here is the most recent picture of Maggie and Monty!"
    await ctx.send(response, file=discord.File(filepath))

bot.run(TOKEN)