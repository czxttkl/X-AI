java -jar shadow.jar 100 gamestate greedymove warrior
java -jar shadow.jar 100 gamestate gamestate warrior "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"

# legend pirate vs. trinity pirate
java -jar shadow.jar 300 greedymove greedymove warrior "14,21,40,12,43,7,23,15,307,109,266,269,265,49,253" "21,12,43,7,23,15,19,307,109,269,265,253,303,256,141"

# legend pirate vs taunt
java -jar shadow.jar 300 greedymove greedymove warrior "14,21,40,12,43,7,23,15,307,109,266,269,265,49,253" "48,36,32,43,24,45,288,11,5,46,225,129,156,228,277"

# legend pirate
"14,21,40,12,43,7,23,15,307,109,266,269,265,49,253"

# trinity pirate
"21,12,43,7,23,15,19,307,109,269,265,253,303,256,141"

# taunt warrior
"48,36,32,43,24,45,288,11,5,46,225,129,156,228,277"

# c'Thun control
"48,5,36,33,43,24,45,46,25,41,225,306,237,58,198"

# use prob_generate pv 8 vs pv 9
java -jar shadow.jar 300 gamestate gamestate warrior "15,45,59,64,122,127,150,198,220,224,229,236,249,256,297" "2,9,32,55,86,100,122,129,152,153,184,267,282,293,302"

java -jar shadow.jar 100 greedymove greedymove warrior "15,45,59,64,122,127,150,198,220,224,229,236,249,256,297" "2,9,32,55,86,100,122,129,152,153,184,267,282,293,302"

