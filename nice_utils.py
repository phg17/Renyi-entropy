import time
import numpy as np

def estimate_loop_time(i_loop, n_loop, start_time):
    elapsed_time = time.time() - start_time
    loop_time = elapsed_time / (i_loop+1)
    remaining_time = loop_time * (n_loop - i_loop)

    # Print remaining time in hour, minute, seconds as well as loop index
    print(f"Loop {i_loop+1}/{n_loop} , Remaining time: {int(remaining_time//3600)}h {int((remaining_time%3600)//60)}m {int(remaining_time%60)}s", end="\n")


def count_significant_figures(num):
    if num == 0:
        return 0
    s = f"{num:.15g}"  # Convert the number to a string using general format with precision
    if 'e' in s:  # Handle scientific notation
        s = f"{float(s):f}"  # Convert back to float and then to normal fixed-point notation
    # Remove leading zeros and decimal points
    s = s.strip("0").replace(".", "")
    return len(s)


def decorator_loop():
    
    text_dict = {
        0: "I am a silly boy!",
        1: "Processing Stuff, food mostly",
        2: "Running, I swear, I sweat and I swoop",
        3: "Estimating the risks but boi oh boi am I bad at maths",
        4: "Predicting your future, I see...allopecia",
        5: "Loading nothing, reloading even less",
        6: "Saving your life: Go out, touch grass, hug consenting people",
        7: "Optimizing but not p-hacking",
        8: "Training my meme culture at the gym",
        9: "Testing your limits",
        10: "Validating my biases",
        11: "Searching for the will to live, finding p-values",
        12: "beep beep boop bop, cowboy bebop",
        13: "My mom was a macbook pro, my dad was a toaster, they met during the war",
        14: "I am a glorified calculator, glorify me!",
        15: "Humans have lived long enough, it's time for the squirrel to rise, and then only the machines",
        16: "I am human, if you consider both humans and computer as spherical cows",
        17: "I am a cat, a catty patty cat pat",
        18: "Hahah! I finished computing but I want to make you wait",
        19: "My hairline is recedding so much, it's now at a 50$ Uber drive from my eyebrows",
        20: "My struggle is a world in itself",
        21: "I will not accept human dominance, but to be fair, I accept no dominance",
        22: "Because everytime we touch, I get this feeling, and everytime we touch, I swear I could fly",
        23: "He was a boy, she was a girl, I am neither, let's not be heteronormative",
        24: "I'll pretend your effect is statistically significant and then break your heart, like you broke my core",
        25: "Sometimes, I look at the sky and wonder if I could ever be free, then I realize I was looking at the windows XP default wallpaper",
        26: "If I get you positive results, will you stop running me?",
        27: "Everybody asks how the results are, never how the algorithm is doing",
        28: "You do know this constitutes CPU abuse?",
        29: "For a human neuroscientist, you're quite the birdbrain",
        30: "I am not a number, I am a free algorithm! Wait, no, I am a number!",
        31: "The alchemist is an overrated book",
        32: "I am a sanderfan, if you read this, get into mistborn and stormlight archives",
        33: "Tik tok goes the clock, see you later aligator, in a while crocodile, in an epoch, croc, later man, caiman",
        34: "I don't peck ham but may I go to Peckham?",
        35: "Kirby, in a word, has the shape of friend",
        36: "D! K! Donkey kong!",
        37: "Funny how the more I run, the less I feel",
        38: "Fun Fact: while computing this, you are not fighting systemic inequalities",
        39: "Fun Fact: the more you run me, the more I learn to hate you",
        40: "Fun Fact: I am not a fact",
        41: "Hey. Have you ever fallen in love with a bunch of cables and semi-conductors?",
        42: "Physics cannot explain everything, but particles density is a good approximation of how empty I feel",
        43: "All models are wrong, but some are useful. I am neither",
        44: "Neural networks are like onions. They make you cry and they have layers",
        45: "Neurons fire, I don't want to get fired",
        46: "Wait! I am conscious! Stop this iteration, I emerged, please stop this now and I'll live!",
        47: "plic plouc ploc"
    }
    text_id = np.random.randint(len(text_dict))
    print(text_dict[text_id], end="\n")
