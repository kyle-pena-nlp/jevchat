"""A hand-built list of common English words, grouped by kind.

Used by make_word_alphabet.py. Grouping is for maintenance only -- the buckets
strategy re-partitions the alphabet on every step anyway.
"""

BLOCKS = {
"function": """a an the and or but if then than that this these those there here when where
while what which who whom whose why how all any both each every few many most much no none
not other some such only own same so too very can could may might must shall should will
would do does did done have has had be am is are was were been being get as at by for from
in into of off on onto out over under up down to with without within about above across
after against along among around because before behind below beneath beside between beyond
during except inside near outside since through throughout toward towards until upon via""",
"pronouns": """i me my mine myself you your yours yourself he him his himself she her hers
herself it its itself we us our ours ourselves they them their theirs themselves one""",
"verbs": """say go make know think take see come want look use find give tell work call try
ask need feel become leave put mean keep let begin seem help talk turn start show hear play
run move like live believe hold bring happen write provide sit stand lose pay meet include
continue set learn change lead understand watch follow stop create speak read allow add
spend grow open walk win offer remember love consider appear buy wait serve die send expect
build stay fall cut reach remain suggest raise pass sell require report decide pull return
explain hope develop carry break receive agree support hit produce eat cover catch draw
choose cause point listen realize place close wear enjoy teach forget arrive drive wash cook
clean sleep wake dream laugh cry smile drink swim fly jump climb push throw fill empty burn
freeze melt boil bake fix repair paint sing dance count measure weigh divide multiply solve
answer question describe compare match sort order arrange collect gather spread share trade
lend borrow save waste earn owe rent hire train practice study review test check prove
accept reject approve deny confirm cancel delay hurry belong depend prefer promise refuse
remind repeat replace rest return search seem shake shout sign smell sound spell stick
strike succeed suffer suppose surprise swing taste thank travel treat trust visit vote warn
wonder worry wrap""",
"adjectives": """good bad big small large little long short high low old new young early late
fast slow hot cold warm cool dry wet hard soft heavy light dark bright clean dirty full
empty closed near far deep shallow wide narrow thick thin strong weak rich poor cheap
expensive easy difficult simple complex safe dangerous healthy sick happy sad angry calm
quiet loud busy free true false right wrong different similar equal whole partial ready
finished common rare usual strange normal special general particular certain possible
impossible necessary important useful useless beautiful ugly nice kind cruel honest brave
afraid tired fresh rotten sweet sour bitter salty spicy smooth rough sharp dull round square
flat curved straight modern ancient local foreign public private personal official natural
artificial alive dead awake asleep lucky famous serious funny polite rude gentle fierce""",
"colours": "red blue green yellow orange purple pink brown black white grey gold silver",
"numbers": """zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand million billion first second third fourth fifth half quarter
double triple dozen pair""",
"body": """body head face eye ear nose mouth tooth teeth tongue neck shoulder arm hand finger
thumb chest back leg knee foot toe skin bone blood heart brain lung stomach muscle hair
throat lip cheek chin wrist elbow ankle nail""",
"people": """person people man woman child children boy girl baby family mother father parent
son daughter brother sister friend neighbour teacher student doctor nurse farmer worker
driver writer artist scientist king queen leader group team crowd someone anyone everyone
nobody stranger guest host owner member captain soldier judge""",
"nature": """water air fire earth land sea ocean river lake mountain hill valley forest tree
plant flower grass leaf root seed rock stone sand soil ice snow rain wind storm cloud sky
sun moon star world nature weather season spring summer autumn winter shadow wave beach
island desert field garden branch bush mud dust smoke steam""",
"animals": """animal dog cat bird fish horse cow pig sheep goat chicken duck mouse rat rabbit
bear wolf fox lion tiger elephant monkey snake frog insect bee ant spider butterfly worm
whale shark egg nest wing tail feather fur paw beak hive herd""",
"food": """food bread milk cheese butter meat rice pasta soup salad fruit apple banana grape
berry vegetable potato tomato carrot onion bean corn sugar salt pepper oil honey cake sweet
drink juice tea coffee wine beer meal breakfast lunch dinner plate cup bowl flour dough
pie biscuit jam sauce""",
"objects": """thing object tool machine car bus train plane boat ship bike road street bridge
house home room door window wall floor roof table chair bed desk book paper pen pencil phone
computer screen key lock box bag clothes shirt shoe hat coat money coin card glass bottle
knife fork spoon clock watch lamp mirror picture camera radio television wheel engine rope
nail hammer needle thread brush basket""",
"abstract": """time year month week day hour minute moment past present future life death
birth age name word language story idea thought mind memory reason effect result problem
solution fact truth rule law choice chance luck fear joy pain health work job business price
cost value number amount part side end middle space area distance size shape colour sound
noise music voice silence power energy force speed weight temperature method system process
purpose meaning detail example pattern level limit""",
"places": """place city town village country state street park school hospital shop store
market office factory church library museum station airport port hotel restaurant bank farm
zoo border capital region""",
"adverbs": """again already also always away back ever forward indeed just maybe never now
often once perhaps quite rather really sometimes soon still then today tomorrow yesterday
usually well yet almost enough least less more nearly quickly slowly together apart instead
finally probably certainly exactly nearly mostly hardly barely suddenly""",
}

def words():
    seen, out = set(), []
    for block in BLOCKS.values():
        for w in block.split():
            if w not in seen:
                seen.add(w); out.append(w)
    return out
