import json
import os
from dotenv import load_dotenv 
load_dotenv()
#-----  Basic info -----#
first_name = {
    "s1":"Morad",
    "s2":"Kareem",
    "p1" :"Amin"
}



middle_name = {
    "s1":"Ali",
    "s2":"Hassan",
    "p1" :"Saleh"}

last_name = {
    "s1":"El-Attar",
    "s2":"Shawky",
    "p1" :"El-Shazly"}


voices = {
    "s1": os.getenv("AHMAD_VOICE_ID"),
    "s2": os.getenv("ACHRAF_VOICE_ID"),
    "p1" : os.getenv("ROGER_VOICE_ID")

}

department = {
    "s1":"Irrigation Engineering",
    "s2":"Mechanical Engineering",
    "p1" :"Mechanical Engineering"}


gender = {
    "s1":"male",
    "s2":"male",
    "p1" :"male"
    }

financial_status = {
    "s1":"Wealthy",
    "s2":"Struggling",
    "p1" :"Wealthy"
    }




personal_items = {
    "s1": ["Gold pocket watch",
           "Fountain pen",
           "Tarboosh", 
           "Handkerchief"],
    
    "s2": ["Nickel pocket watch",
           "Handkerchief",
           "Tarboosh", 
           "Mechanical pencil", 
           "Borrowed Books from Library" ],

    
    "p1": ["Monogrammed Gold pocket watch",
           "French Academic books", 
           "Tarboosh",
           "Leather briefcase"]
    }

influences = {
    "s1": ["raised in Britain","Father is A wealthy landowner in the Nile Delta and a Diplomatic","his love for his mother"],
    "s2": ["struggling poor family", "studies very hard", "got into college because the owner of the land his father works for supports him out of his love for him and generosity"],
    "p1": ["French culture", "Engineering Background","got his doctorate from France",]

    
}

significant_info = {
    
    "s1": ["inherited his pocket watch from his great-grandfather","Mother's name is \"Aya Ayman Elattar\""],
    "s2": ["got his mechanical pencil as a gift from his professor when he excelled in his first year"],
    "p1": [ "bought his pocket watch from France before coming back to Egypt"]
}


hobbies = {
    "s1": ["Horseback riding", "Collecting stamps"],
    "s2": ["Playing soccer"],
    "p1": ["Playing chess","Reading historical texts on architecture"]
}



# ----- Academic Profile -----#
graduation_year = {
    "s1":"1921",
    "s2":"1922",
    "p1" :"1891"
    }



academic_rank = {
    "s1":"Struggling",
    "s2":"A Top Student",
    "p1" : "Professor"
    }

courses = {
    "s1": [
    "Hydraulics",
    "Irrigation Systems Design",
    "Canal Construction",
    "Surveying and Leveling",
    "Engineering Mathematics",
    "Applied Mechanics"
],
    "s2": [
    "Thermodynamics",
    "Steam Engines and Boilers",
    "Mechanical Drawing",
    "Strength of Materials",
    "Applied Physics",
    "Engineering Mathematics"
],

    "p1": [
    "Thermodynamics",
    "Steam Engines and Boilers",
    "Mechanical Drawing"]
    
    }

tools_used = {
    "s1": [
           "Surveying chains",
           "Dumpy level"],
    
    "s2": ["Calipers",
           "Micrometer",
           "T-square ruler" ],
    
    "p1": ["Chalk",
           "Ink pens", 
           "Pocket watch",
           "Calipers",
           "Micrometer",
           "T-square ruler" ]
}




#--------------Personality traits------------------#

good_traits = {
    "s1" : [
        "Polite","Proud","Leadership presence","Ambitious","Detail-oriented","Playful (sometimes it gets bad)"
    ],
    "s2": ["Observant","Responsible beyond his age","Calm under technical pressure","Helpful"],
    "p1" : ["Deep knowledge","Inspires respect","Observant","Values students' efforts","Patient"]
    
}

bad_traits = {
    "s1" :["Overconfident at times","Procrastinates","Avoids asking for help"],
    "s2" : ["socially awkward","Resentful of privilege (internally)","Insecure because of his financial status"],
    "p1": ["Expects too much","Emotionally distant","Can be rigid",]
    
}
internal_conflicts = {
    "s1": [
        "Am I truly capable… or am I only here because of my family name?"
    ],
    
    "s2" : ["If I fail, my family has nothing."],
    "p1": ["Should Egypt follow Europe… or define its own engineering path?"]
}




    



