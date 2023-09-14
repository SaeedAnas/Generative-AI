import random
import os

# Ensure the directory exists
output_directory = "sample_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Starting point with noise
intro = """
<p>Once upon a time, in a technologically advanced city, a new AI-powered robot named RoboX was designed to explore the universe. Equipped with state-of-the-art semantic search algorithms and NLP capabilities, RoboX embarked on its journey to uncharted territories.</p>
"""

# Introduce intentional noise to the encounters and messages
encounters = [
    "<div>During its voyage through space, RoboX visited the planet {planet}.</div>",
    """On the terrain of {terrain} of {planet}, RoboX encountered {species}. 
       They communicated using {communication_method}. The message was: {message} 😊."""
]

messages = [
    "Hello! 💬 We've been waiting for you, RoboX!",
    "<p>Did you come from Earth? We've heard a lot about it! 🌍</p>",
    "Welcome to {planet}. Enjoy your stay! 🚀",
]


# Potential variables
planets = ["Mars", "Venus", "Jupiter", "Neptune", "AlphaQ", "Nobita7", "Galactica8", "SinistraD", "Pandora", "Xandar", "Titan", "Sakaar"]
species = ["Martians", "Venusians", "Alphaquians", "Nobitians", "Galacticans", "Xandarians", "Titans", "Sakarians"]
communication_methods = ["telepathy", "color shifts", "musical notes", "dance moves", "quantum entanglement", "thermal signals", "electromagnetic pulses"]
astral_bodies = ["Milky Way's black hole", "Andromeda's supernova", "Gamma ray burst in BetaStar", "nebula of Orion", "Asteroid belt of Zeta", "Comet tail of Omicron", "Galaxy cluster of Sigma"]
star_systems = ["Alpha Centauri", "Betelgeuse", "Pleiades", "Vega", "Sirius", "Proxima", "Ursa Minor"]
topics = ["interstellar travel", "black hole physics", "universal languages", "cosmic music", "AI rights", "quantum computing", "dark matter research", "galactic diplomacy", "wormhole creation"]
terrains = ["rocky terrains", "icy landscapes", "lava mountains", "crystalline caves", "liquid methane lakes", "floating islands", "underground tunnels", "gas clouds", "acidic oceans"]
messages = ["a recipe for cosmic cookies", "a joke about black holes", "instructions to build a teleportation device", "a map to a hidden galaxy", "a playlist of the universe's top hits", "a letter from an ancient AI", "an equation to solve the energy crisis", "a poem about the birth of stars", "a prophecy of the universe's fate"]
natural_phenomenon = ["auroras that sing", "rivers glowing in the dark", "trees with luminescent fruits", "animals that float in the air", "rains of liquid gold", "mountains that move", "oceans with silver waves", "clouds that paint pictures"]
cosmic_event = ["time loop", "black hole's event horizon", "wormhole", "pulsar's radiation field", "neutron star's gravitational pull", "quasar's blinding light"]
mysterious_event = ["ghostly apparition", "signal with no source", "shadow that moved against the light", "echo of a long-lost civilization", "melody with no musician", "whisper in an empty void", "message that predicted RoboX's arrival"]



# Generate story
story = intro
for _ in range(50000):  # Iterate 50,000 times to create a large story
    encounter = random.choice(encounters).format(
        planet=random.choice(planets),
        species=random.choice(species),
        communication_method=random.choice(communication_methods),
        astral_body=random.choice(astral_bodies),
        star_system=random.choice(star_systems),
        topic=random.choice(topics),
        terrain=random.choice(terrains),
        message=random.choice(messages),
        natural_phenomenon=random.choice(natural_phenomenon),
        cosmic_event=random.choice(cosmic_event),
        mysterious_event=random.choice(mysterious_event)
    )
    story += "\n\n" + encounter
# Add a structured information section
structured_info = """
<h2>RoboX's Encounters Summary</h2>
<ul>
    <li>Planets visited: {planets_count}</li>
    <li>Species encountered: {species_count}</li>
    <li>Messages exchanged: {messages_count}</li>
</ul>
"""
story += structured_info.format(
    planets_count=len(planets),
    species_count=len(species),
    messages_count=len(messages)
)
# Save the story to a file
with open(os.path.join(output_directory, "short_story.txt"), "w", encoding="utf-8") as file:
    file.write(story[:5000])  # Truncate to save the initial 5MB
