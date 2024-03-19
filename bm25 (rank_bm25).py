from rank_bm25 import BM25Okapi

# Create corpus of documents
documents = [
    """
        The Lovecraftian world of Arkham Horror: The Card Game is the setting for a supportive living card game. New content, including investigators, cards, and a brand-new scenario, has been added to the game with the release of the Fortune & Folly Scenario Pack.

        New to the game is "The Night's Usurper," set in a museum, and it is part of the Fortune & Folly Scenario Pack. In order to defeat a sinister force and save the night, players must investigate the displays. Players perform actions and draw encounter cards that can impede their progress one by one, just like in a regular game of Arkham Horror: The Card Game. A new Exhibit deck and a token representing an investigator's Night Terrors are introduced in this scenario.

        Aesthetically, the Fortune & Folly Situation Pack boasts stunning illustrations that perfectly capture the dark, foreboding tone of <PERSON>'s canon. The museum is depicted vividly, and each display has its own personality and history. The stunning artwork of the new detectives offers novel interpretations of classic characters.

        Adding the Fortune & Folly Situation Pack to your Card Game collection is a fun way to expand your game. The new scenario's design and difficulty will appeal to seasoned players, and the addition of investigators and additional cards expands their strategic options. Compared to other Lovecraftian settings, the museum is a welcome change of pace, and the Night Terrors mechanic introduces an exciting new layer of tension.

        Like all expansions for Arkham Horror: has a lot of potential for playthroughs. Because the new scenario has random elements and branching paths, no two runs will be the same. There are even more opportunities for experimentation and creativity with the new investigators and cards.

        The Fortune & Folly Scenario Pack's new scenario is well-designed and offers a new challenge. Additionally, the new investigators and cards give players more control over how their decks are constructed. The museum setting is even more refreshing than the typical Lovecraftian locations. The new content is great, but it does require the base game and any expansions to play.

        You can play the Fortune & Folly Scenario Pack by yourself or with up to three other players in a co-op mode. Both single-player and multiplayer games can profit from the new situation, and solo players will see the value in the new specialists for the assortment they add to deck-building. When working together to solve puzzles and defeat threats, players in a multiplayer game can benefit from each other's strategic input.

    """,
    """
        Hello today we would be reviewing another great board game that's called The Great Zimbabwe, it's an interesting board game which was created and designed by a great man called <PERSON>. The Great Zimbabwe is an amazing board which can be played by 1-4 players, Here the players take on two roles which is that of a trader and that of a craftsman which takes place in an ancient city of Great Zimbabwe where their try to get wealth and prestige, this board game is kind of different from other board game because it's board is modular that is different each time you play it and it also has many different types of building which you can easily upgrade and construct.

        The Great Zimbabwe board game has many series of round which you can play on the board where different players can place their different workers on different locations on the board and on each location the players place their workers it has different series of action that can be performed such as building structures, collecting different resources and also trading with other players. Interesting enough players can also get special abilities by using their workers to gods.

        For the game to end a player must reach or achieve a certain number of victory points and also the game can end when the central market has finished, then the player with the highest victory points wins at the end of the games.

        Their are many interesting features of The Great Zimbabwe board game that attracts many players such as it's very challenging and strategic, it has a very strong theme, it has beautiful artwork, it also has an interesting well written rule book, But as interesting as The Great Zimbabwe board game is it still has its cons which are noticable such as it's difficult to learn for some players, their many rules and information to follow and keep track of, it has a very long playtime such as you need to play the game for a very long time before it ends and also some times the gameplay is very stressing.

        Even with all it's cons,The Great Zimbabwe game is a very interesting game that offers players in depth strategy and is also very challenging. Its a really good board game that's one of a kind and it would be worth checking out.
        
    
    """,
    """
    How is the weather today?
    
    """
    ]

query = """
    Agricola is a worker placement game for 1 to 5 players, with a focus on resource management. It is a subsistence farming simulation game, set in the 17th century.

    Place the central board on the table, set up all the other cards as described in the game rules. There are wooden room tiles, clay room tiles, stone room tiles, field tiles, food tokens, begging cards, goods pieces, livestock pieces, action cards, minor and major improvement cards and occupation cards.

    Each player starts with 2 workers, a farmer and his wife, a small wooden house, a board to fill with cultivated fields, houses, or barns with fences and animals, 7 Occupation cards and 7 Minor Improvement cards. A player can take 2 actions per round.

    The game lasts 14 rounds, divided into 6 stages. After rounds 4, 7, 9, 11, 13 and 14, a harvest occurs, where each player must feed their family or pay the consequence by taking a beggar’’s card which makes them lose 3 points at the end of the game.

    Each round begins by flipping the next round card, which offers a new action for the players. Next, all action spaces with a red arrow are replenished, for example, the wood hut gets 3 wood pieces per round which add up if no one has claimed them. Now each player takes turns placing a worker on an action space and performing the action, until all workers have been placed. Only one person can be placed on any action space at a time. Actions include building fences, installing barns, collecting goods such as clay and wood, improving your house, plowing fields, having babies, breeding animals and growing food. When all workers have been placed and actions taken, players recover their workers and a new round begins, unless it’’s harvest time. At the end of the 14 rounds, the player with the most points is the best farmer and wins.

    The game is attractive, the tokens are made of wood and are easy to manipulate. The actions are simple to understand, with many options to pursue. The game has great replay value because so many different strategies can be used. It’’s a most interesting game and well worth trying for board game night.
    

    """

# Tokenize each document
tokenized_doc = [doc.lower().split() for doc in documents] # 假设分词并转换为小写

# Create a BM25 index from the tokenized document corpus
bm25 = BM25Okapi(tokenized_doc)

# Query the BM25 index
tokenized_query = query.lower().split()

# Calculate BM25 scores for the query
doc_scores = bm25.get_scores(tokenized_query)

# Retrieve documents sorted by their BM25 score in relation to the query
# sorted_docs = bm25.get_top_n(tokenized_query, documents, n=len(documents))
# print("sorted_docs: ", sorted_docs)

bm25_sorted = [[doc_scores[i], documents[i][:50]] for i in range(len(documents))]
bm25_sorted = sorted(bm25_sorted, reverse=True)

# Print BM25 scores and corresponding documents
print("BM25 scores and corresponding documents:")
for i in bm25_sorted:
    print("i: ", i)




# from rank_bm25 import BM25Okapi
# import pickle
# import re


# # sample document
# # documents = [
# #     "the sky is blue",
# #     "the sun is bright",
# #     "the sun in the sky is bright",
# #     "we can see the shining sun, the bright sun"
# # ]

# def preprocess(text):
#     # remove punctuation, turn into lower case, and tokenize
#     text = re.sub(r'\W', ' ', text)
#     text = text.lower()
#     return text.split()

# documents = [
#     """
#     The Lovecraftian world of Arkham Horror: The Card Game is the setting for a supportive living card game. New content, including investigators, cards, and a brand-new scenario, has been added to the game with the release of the Fortune & Folly Scenario Pack.

#     New to the game is "The Night's Usurper," set in a museum, and it is part of the Fortune & Folly Scenario Pack. In order to defeat a sinister force and save the night, players must investigate the displays. Players perform actions and draw encounter cards that can impede their progress one by one, just like in a regular game of Arkham Horror: The Card Game. A new Exhibit deck and a token representing an investigator's Night Terrors are introduced in this scenario.

#     Aesthetically, the Fortune & Folly Situation Pack boasts stunning illustrations that perfectly capture the dark, foreboding tone of <PERSON>'s canon. The museum is depicted vividly, and each display has its own personality and history. The stunning artwork of the new detectives offers novel interpretations of classic characters.

#     Adding the Fortune & Folly Situation Pack to your Card Game collection is a fun way to expand your game. The new scenario's design and difficulty will appeal to seasoned players, and the addition of investigators and additional cards expands their strategic options. Compared to other Lovecraftian settings, the museum is a welcome change of pace, and the Night Terrors mechanic introduces an exciting new layer of tension.

#     Like all expansions for Arkham Horror: has a lot of potential for playthroughs. Because the new scenario has random elements and branching paths, no two runs will be the same. There are even more opportunities for experimentation and creativity with the new investigators and cards.

#     The Fortune & Folly Scenario Pack's new scenario is well-designed and offers a new challenge. Additionally, the new investigators and cards give players more control over how their decks are constructed. The museum setting is even more refreshing than the typical Lovecraftian locations. The new content is great, but it does require the base game and any expansions to play.

#     You can play the Fortune & Folly Scenario Pack by yourself or with up to three other players in a co-op mode. Both single-player and multiplayer games can profit from the new situation, and solo players will see the value in the new specialists for the assortment they add to deck-building. When working together to solve puzzles and defeat threats, players in a multiplayer game can benefit from each other's strategic input.

#     """,
#     """
#     Hello today we would be reviewing another great board game that's called The Great Zimbabwe, it's an interesting board game which was created and designed by a great man called <PERSON>. The Great Zimbabwe is an amazing board which can be played by 1-4 players, Here the players take on two roles which is that of a trader and that of a craftsman which takes place in an ancient city of Great Zimbabwe where their try to get wealth and prestige, this board game is kind of different from other board game because it's board is modular that is different each time you play it and it also has many different types of building which you can easily upgrade and construct.

#     The Great Zimbabwe board game has many series of round which you can play on the board where different players can place their different workers on different locations on the board and on each location the players place their workers it has different series of action that can be performed such as building structures, collecting different resources and also trading with other players. Interesting enough players can also get special abilities by using their workers to gods.

#     For the game to end a player must reach or achieve a certain number of victory points and also the game can end when the central market has finished, then the player with the highest victory points wins at the end of the games.

#     Their are many interesting features of The Great Zimbabwe board game that attracts many players such as it's very challenging and strategic, it has a very strong theme, it has beautiful artwork, it also has an interesting well written rule book, But as interesting as The Great Zimbabwe board game is it still has its cons which are noticable such as it's difficult to learn for some players, their many rules and information to follow and keep track of, it has a very long playtime such as you need to play the game for a very long time before it ends and also some times the gameplay is very stressing.

#     Even with all it's cons,The Great Zimbabwe game is a very interesting game that offers players in depth strategy and is also very challenging. Its a really good board game that's one of a kind and it would be worth checking out.
#     """

# ]

# queries = [
#     """
#     Agricola is a worker placement game for 1 to 5 players, with a focus on resource management. It is a subsistence farming simulation game, set in the 17th century.

#     Place the central board on the table, set up all the other cards as described in the game rules. There are wooden room tiles, clay room tiles, stone room tiles, field tiles, food tokens, begging cards, goods pieces, livestock pieces, action cards, minor and major improvement cards and occupation cards.

#     Each player starts with 2 workers, a farmer and his wife, a small wooden house, a board to fill with cultivated fields, houses, or barns with fences and animals, 7 Occupation cards and 7 Minor Improvement cards. A player can take 2 actions per round.

#     The game lasts 14 rounds, divided into 6 stages. After rounds 4, 7, 9, 11, 13 and 14, a harvest occurs, where each player must feed their family or pay the consequence by taking a beggar’’s card which makes them lose 3 points at the end of the game.

#     Each round begins by flipping the next round card, which offers a new action for the players. Next, all action spaces with a red arrow are replenished, for example, the wood hut gets 3 wood pieces per round which add up if no one has claimed them. Now each player takes turns placing a worker on an action space and performing the action, until all workers have been placed. Only one person can be placed on any action space at a time. Actions include building fences, installing barns, collecting goods such as clay and wood, improving your house, plowing fields, having babies, breeding animals and growing food. When all workers have been placed and actions taken, players recover their workers and a new round begins, unless it’’s harvest time. At the end of the 14 rounds, the player with the most points is the best farmer and wins.

#     The game is attractive, the tokens are made of wood and are easy to manipulate. The actions are simple to understand, with many options to pursue. The game has great replay value because so many different strategies can be used. It’’s a most interesting game and well worth trying for board game night.
#     """
# ]

# # tokenize document
# tokenized_docs = [preprocess(doc) for doc in documents]
# # print("tokenized_docs: ", tokenized_docs)
# bm25 = BM25Okapi(tokenized_docs)
# # print("bm25: ", bm25)

# # calculate bm25 score for each query
# bm25_scores = {query: bm25.get_scores(preprocess(query)) for query in queries}
# print("bm25_scores: ", bm25_scores)

# # with open('bm25_scores.pkl', 'wb') as f:
# #     print("bm25_scores, f: ", bm25_scores, f)
# #     pickle.dump(bm25_scores, f)