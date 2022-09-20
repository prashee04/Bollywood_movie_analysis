
import nltk
import random
import pandas as pd
import numpy as np
from nltk.corpus import names


Male_Names = names.words('male.txt')
Female_Names = names.words('female.txt')


Name_List = [(Name , 'male') for Name in Male_Names] + [(Name , 'female') for Name in Female_Names]

random.shuffle(Name_List)


def Name_Feature (name):
    return {
        'last_char': name[-1],
        'last_two_char': name[-2:],
        'last_three_char': name[-3:],
        'first_char': name[0],
        'first_two_char': name[:2],
        'first_three_char': name[:3]
    }

features = [ (Name_Feature(name= NAME), Gender) for (NAME, Gender) in Name_List ]

Training_Set = features[ :round(len(features) * .7)]
Testing_Set = features[round(len(features) * .7): ]

classifier = nltk.NaiveBayesClassifier.train(Training_Set)

round(nltk.classify.accuracy(classifier , Testing_Set) * 100, 2)

Final_Classifier_Model = nltk.NaiveBayesClassifier.train(features)

import pickle
import os

with open('Final_Classifier_Model.pkl', 'wb') as fileWriteStream:
    pickle.dump(Final_Classifier_Model, fileWriteStream)
    fileWriteStream.close()

def Identify_Person_Gender(Text):
    import re
    import pandas as pd
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize

    import pickle
    with open('Final_Classifier_Model.pkl', 'rb') as fileReadStream:
        Prediction_Model=pickle.load(fileReadStream)
        fileReadStream.close() 
        
    Final_Name= []
    Final_Gender= []

    Sent_List = sent_tokenize(Text.title())

    for sent in Sent_List:
        word_list = word_tokenize(sent)

        pos_tags = nltk.pos_tag(word_list)

        Person_Name= [name[0] for name in pos_tags if name[1] in ['NNP','NN']]
        Person_Name= re.sub(r'[^a-z A-Z . ,]',r'',str(Person_Name))

        Person_Name_feature= {  'last_char': Person_Name[-1],
                                'last_two_char': Person_Name[-2:],
                                'last_three_char': Person_Name[-3:],
                                'first_char': Person_Name[0],
                                'first_two_char': Person_Name[:2],
                                'first_three_char': Person_Name[:3] 
                             }

        Gender_Prediction=Prediction_Model.classify(Person_Name_feature)
        
        Final_Name.append(Person_Name)
        Final_Gender.append(Gender_Prediction)        
    
    prediction_result= pd.DataFrame({'Name':Final_Name, 'Gender': Final_Gender})

    return(prediction_result)

text = """Arun Neelakandan, a teenager from Kerala, joins KC Tech, a popular engineering college in Chennai for his graduation. He falls in love with college sweetheart Darshana at the first sight. They grow closer after Arun is ragged by his seniors, but a few scuffles and incidents get the seniors suspended and they start dating. Arun accompanies his friend Antony to meet his online girlfriend, and feels attracted to her colleague. He lies to her, states he is single and they lean in for a kiss but they are interrupted by moral police. A guilt-ridden Arun confesses to Darshana, who lividly calls off the relationship. In the heat of their argument, they challenge each other that they will have other romantic partners better than each other.Arun's life goes downhill after the breakup. He takes to ragging juniors and thrashing up other people and outsiders in his second year and is addicted to alcohol. He begins a relationship with Maya but doesn't find himself happy. Meanwhile, Darshana starts dating Kedar, a womanizer. He warns Darshana but she ignores it as a sign of jealousy. However, Arun is proven right and Darshana slaps Kedar. Kedar threatens to spread rumours about her, and her friend convey this to and Arun beats Kedar up and silences him. At home, Arun's parents notice something is wrong with their son. His father advices him to quit alcohol and turn over a new leaf in his life.Arun moves out of his shared hostel room to a more peaceful atmosphere. He joins their classmate Selva's coaching classes along with Antony and they soon rebound in academics, with Arun being fifth in class, and both clearing many of their supplies. Later, Darshana too joins Selva's coaching class and finds herself liking the newly changed Arun. Selva dies in a bus accident and everyone in his class is grief-stricken. Maya calls off her relationship with Arun after her father's death, realizing that Arun does not love her but only wanted her to make Darshana jealous. On the last day of college, all students are led into a room called "the secret alley," (which was shown in their first year and they learned that only the final year students can see what it is on their final day) where they leave a message for the next batch of students. However, Arun finds himself unable to write anything. Darshana, who accompanies Arun to his train, asks him if they would have still been together if not for the argument four years ago. He does not have an answer.After graduation, Arun gets a job in a campus interview. Two years later, Arun feels dissatisfied with his job and leaves it. Darshana, now a YouTube vlogger, advises him to pursue his dreams. On a bus journey, he meets Jimmy, a wedding photographer who is in need of a partner, drags him to photography. Arun decides that they should be a company that specialises in intimate weddings since no such brand currently exists in Kerala. He gets their company a shoutout from Prateek Tiwari, his batchmate at KC Tech who, is now a popular Bollywood singer. After the shoutout, they get lots of bookings and enquiries and they become famous and Arun now feels satisfied with what he is doing.During one of such weddings, Arun sees Nithya Balagopal and feels instantly attracted to her. Impressed with his photographs, Nithya recommends Arun to her cousin, who is about to be married. However, Arun discovers that the groom is Kedar and exposes his character to Nithya. The wedding is called off and Nithya is grateful to Arun for saving her cousin's life. They grow closer and Nithya accompanies Arun to Chennai, for a wedding and back to the terrace where Selva held his coaching classes, but unfortunately now locked up after arrival of new tenants.With their parents' approval, Nithya and Arun fix their marriage. Darshana attends the wedding reception but finds herself unable to come to terms with the fact Arun can no longer be hers and leaves in tears. Arun and Nithya get married and live happily thereafter. Three years later, Nithya gives birth to a baby boy. A joyous Arun feels like telling Darshana the news first.Darshana's marriage is fixed, and Arun and Nitya along with their son go to her place for the same, but Darshana, who still has feelings for Arun, acts too close around him, which irks Nithya. That night, she calls her to meet at the same place they confessed to each other, the beach. Arun tries to secretly leave but Nithya wakes up and he lies to her saying that they have a bachelorette party for her and leaves. But Arun feels guilty and goes to the room to find Nithya in a sad state realising that he lied. He confesses that he is going to meet Darshana. She allows him and she goes back to sleep peacefully. Darshana, who is due to get married the next day, expresses her reservations about marriage to Arun, and asks him once again if they would have still been together were it not for the argument. Arun tells her to stop thinking about the what-ifs and look forward to her future and they tearfully embrace. Darshana gets married.While in Chennai, Arun takes a detour to KC Tech and gets the key to the secret alley. He writes a thank you note to the college for making him who he is and exits the campus, where he, Nithya and their son return home."""

#tab= Identify_Person_Gender(text)

import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(text)
d=[]
for ent in doc.ents:
    
    d.append(ent.text)

#tab = Identify_Person_Gender()

#fdf = tab[tab['Gender'] == 'female' ]

#mdf = tab[tab['Gender']== 'male']

def gender_bias(a):
    tab= Identify_Person_Gender(a)
    res = tab.groupby('Gender').size().reset_index(name='counts')
    return res

gender_bias("""Arun Neelakandan, a teenager from Kerala, joins KC Tech, a popular engineering college in Chennai for his graduation. He falls in love with college sweetheart Darshana at the first sight. They grow closer after Arun is ragged by his seniors, but a few scuffles and incidents get the seniors suspended and they start dating. Arun accompanies his friend Antony to meet his online girlfriend, and feels attracted to her colleague. He lies to her, states he is single and they lean in for a kiss but they are interrupted by moral police. A guilt-ridden Arun confesses to Darshana, who lividly calls off the relationship. In the heat of their argument, they challenge each other that they will have other romantic partners better than each other.Arun's life goes downhill after the breakup. He takes to ragging juniors and thrashing up other people and outsiders in his second year and is addicted to alcohol. He begins a relationship with Maya but doesn't find himself happy. Meanwhile, Darshana starts dating Kedar, a womanizer. He warns Darshana but she ignores it as a sign of jealousy. However, Arun is proven right and Darshana slaps Kedar. Kedar threatens to spread rumours about her, and her friend convey this to and Arun beats Kedar up and silences him. At home, Arun's parents notice something is wrong with their son. His father advices him to quit alcohol and turn over a new leaf in his life.Arun moves out of his shared hostel room to a more peaceful atmosphere. He joins their classmate Selva's coaching classes along with Antony and they soon rebound in academics, with Arun being fifth in class, and both clearing many of their supplies. Later, Darshana too joins Selva's coaching class and finds herself liking the newly changed Arun. Selva dies in a bus accident and everyone in his class is grief-stricken. Maya calls off her relationship with Arun after her father's death, realizing that Arun does not love her but only wanted her to make Darshana jealous. On the last day of college, all students are led into a room called "the secret alley," (which was shown in their first year and they learned that only the final year students can see what it is on their final day) where they leave a message for the next batch of students. However, Arun finds himself unable to write anything. Darshana, who accompanies Arun to his train, asks him if they would have still been together if not for the argument four years ago. He does not have an answer.After graduation, Arun gets a job in a campus interview. Two years later, Arun feels dissatisfied with his job and leaves it. Darshana, now a YouTube vlogger, advises him to pursue his dreams. On a bus journey, he meets Jimmy, a wedding photographer who is in need of a partner, drags him to photography. Arun decides that they should be a company that specialises in intimate weddings since no such brand currently exists in Kerala. He gets their company a shoutout from Prateek Tiwari, his batchmate at KC Tech who, is now a popular Bollywood singer. After the shoutout, they get lots of bookings and enquiries and they become famous and Arun now feels satisfied with what he is doing.During one of such weddings, Arun sees Nithya Balagopal and feels instantly attracted to her. Impressed with his photographs, Nithya recommends Arun to her cousin, who is about to be married. However, Arun discovers that the groom is Kedar and exposes his character to Nithya. The wedding is called off and Nithya is grateful to Arun for saving her cousin's life. They grow closer and Nithya accompanies Arun to Chennai, for a wedding and back to the terrace where Selva held his coaching classes, but unfortunately now locked up after arrival of new tenants.With their parents' approval, Nithya and Arun fix their marriage. Darshana attends the wedding reception but finds herself unable to come to terms with the fact Arun can no longer be hers and leaves in tears. Arun and Nithya get married and live happily thereafter. Three years later, Nithya gives birth to a baby boy. A joyous Arun feels like telling Darshana the news first.Darshana's marriage is fixed, and Arun and Nitya along with their son go to her place for the same, but Darshana, who still has feelings for Arun, acts too close around him, which irks Nithya. That night, she calls her to meet at the same place they confessed to each other, the beach. Arun tries to secretly leave but Nithya wakes up and he lies to her saying that they have a bachelorette party for her and leaves. But Arun feels guilty and goes to the room to find Nithya in a sad state realising that he lied. He confesses that he is going to meet Darshana. She allows him and she goes back to sleep peacefully. Darshana, who is due to get married the next day, expresses her reservations about marriage to Arun, and asks him once again if they would have still been together were it not for the argument. Arun tells her to stop thinking about the what-ifs and look forward to her future and they tearfully embrace. Darshana gets married.While in Chennai, Arun takes a detour to KC Tech and gets the key to the secret alley. He writes a thank you note to the college for making him who he is and exits the campus, where he, Nithya and their son return home.""")