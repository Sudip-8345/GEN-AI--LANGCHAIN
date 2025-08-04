from langchain.text_splitter import RecursiveCharacterTextSplitter

text = '''After running for a while, the hare got bored.  He felt bad, because he challenged the tortoise
 to a race when everyone knows tortoises are one of the slowest animals in the world.  To keep
 things interesting, he sat down under a tree to take a nap and wait for the tortoise to catch up.
 The tortoise was slow, as tortoises always have been.  Little by little, the tortoise ran down
 the road.  She ran at a slow but steady pace, and she kept her speed even.
 After ten minutes, the tortoise passed the tree where the hare sat sleeping.  The tortoise was
 focused on the race, though, and she didn’t even see him there.  Five minutes later, the tortoise
 crossed the finish line.
 “Congratulations!” said the fox.  “You came in first place.”
 Right then, the hare came running up the road.  He woke up from his nap, but it was too late.
 The fastest runner doesn’t always win the race.'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0
)
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[-2:])