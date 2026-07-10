"""Cook the analogies WordNet does not have.

The fan-in analysis says only relations that DETERMINE their target add
derivable words. `aachen -> city` has fan-in 10; it makes `aachen` a source and
derives nothing. So we cook exactly two kinds:

1. DERIVATIONAL MORPHOLOGY (a base determines its derived form uniquely).
   WordNet records `derivationally_related` as an undirected soup with fan-in
   1.21 and no morphological process attached. We recover the processes as
   separate, directed, near-bijective relations: `happy -> happiness`,
   `nation -> national`, `dark -> darken`. Every rule handles English
   allomorphy (e-deletion, y->i, consonant doubling) and a pair is emitted only
   when BOTH forms are attested in the 38k vocabulary. Nothing is invented.

2. BIJECTIVE ENCYCLOPEDIC FACTS. These are what BATS has and WordNet lacks:
   `country -> capital`, `country -> demonym`, `male -> female`,
   `animal -> young`, `animal -> sound`, `number -> ordinal`. Each is a
   function: one source, one answer. These are facts, written out and filtered
   to the vocabulary; a wrong pair costs precision and is visible in the
   held-out score.

Relations with high fan-in (`name -> nationality`, `thing -> colour`,
`city -> county`) are deliberately NOT cooked: they cannot derive their target.
"""
import json, collections, itertools

D = "real/english"
OUT = f"{D}/cooked.json"

VOWELS = set("aeiou")


def _e_drop(w):
    return w[:-1] if w.endswith("e") else w


def _y_to_i(w):
    return w[:-1] + "i" if w.endswith("y") and len(w) > 2 and w[-2] not in VOWELS else w


def _double(w):
    """run -> runn (CVC ending, not w/x/y)"""
    if len(w) >= 3 and w[-1] not in "wxy" and w[-1] not in VOWELS \
            and w[-2] in VOWELS and w[-3] not in VOWELS:
        return w + w[-1]
    return w


# base -> derived. Each returns a LIST of candidate surface forms; the first
# that is attested in the vocabulary wins. Direction matters: the base
# determines the derived form, so the DERIVED word is the derivable one.
SUFFIX = {
    "ness":   lambda w: [_y_to_i(w) + "ness", w + "ness"],
    "ly":     lambda w: [_y_to_i(w) + "ly", w + "ly"],
    "er_agent": lambda w: [w + "r" if w.endswith("e") else _double(w) + "er"],
    "ment":   lambda w: [w + "ment"],
    "able":   lambda w: [_e_drop(w) + "able", w + "able"],
    "tion":   lambda w: [_e_drop(w) + "ation", _e_drop(w) + "ion", w + "tion"],
    "ist":    lambda w: [_e_drop(w) + "ist", w + "ist"],
    "ism":    lambda w: [_e_drop(w) + "ism", w + "ism"],
    "ic":     lambda w: [_e_drop(w) + "ic", w + "ic"],
    "ical":   lambda w: [_e_drop(w) + "ical", w + "ical"],
    "al":     lambda w: [_e_drop(w) + "al", w + "al"],
    "ous":    lambda w: [_e_drop(w) + "ous", w + "ous"],
    "ful":    lambda w: [_y_to_i(w) + "ful", w + "ful"],
    "less":   lambda w: [_y_to_i(w) + "less", w + "less"],
    "ify":    lambda w: [_y_to_i(w) + "fy", _e_drop(w) + "ify"],
    "ize":    lambda w: [_e_drop(w) + "ize", w + "ize"],
    "ization": lambda w: [_e_drop(w) + "ization"],
    "en_verb": lambda w: [w + "en"],
    "hood":   lambda w: [w + "hood"],
    "ship":   lambda w: [w + "ship"],
    "dom":    lambda w: [w + "dom"],
    "ity":    lambda w: [_e_drop(w) + "ity", w + "ity"],
    "ance":   lambda w: [_e_drop(w) + "ance", _e_drop(w) + "ence"],
    "y_adj":  lambda w: [_double(w) + "y", w + "y"],
    "ish":    lambda w: [w + "ish"],
    "like":   lambda w: [w + "like"],
    "ward":   lambda w: [w + "ward"],
    "wise":   lambda w: [w + "wise"],
}

PREFIX = {
    "un":   lambda w: ["un" + w],
    "in":   lambda w: ["in" + w],
    "im":   lambda w: ["im" + w],
    "non":  lambda w: ["non" + w],
    "dis":  lambda w: ["dis" + w],
    "mis":  lambda w: ["mis" + w],
    "re":   lambda w: ["re" + w],
    "pre":  lambda w: ["pre" + w],
    "post": lambda w: ["post" + w],
    "over": lambda w: ["over" + w],
    "under": lambda w: ["under" + w],
    "sub":  lambda w: ["sub" + w],
    "super": lambda w: ["super" + w],
    "anti": lambda w: ["anti" + w],
    "inter": lambda w: ["inter" + w],
    "co":   lambda w: ["co" + w],
    "counter": lambda w: ["counter" + w],
    "semi": lambda w: ["semi" + w],
    "micro": lambda w: ["micro" + w],
    "multi": lambda w: ["multi" + w],
}

# ---- bijective encyclopedic facts (BATS-style; WordNet has none of these) ----
CAPITAL = """afghanistan kabul|albania tirana|algeria algiers|angola luanda|argentina
buenos|armenia yerevan|austria vienna|azerbaijan baku|bahamas nassau|bangladesh dhaka|
belarus minsk|belgium brussels|bolivia sucre|botswana gaborone|brazil brasilia|bulgaria
sofia|cambodia phnom|canada ottawa|chile santiago|china beijing|colombia bogota|croatia
zagreb|cuba havana|denmark copenhagen|ecuador quito|egypt cairo|estonia tallinn|ethiopia
addis|finland helsinki|france paris|gabon libreville|georgia tbilisi|germany berlin|ghana
accra|greece athens|guinea conakry|hungary budapest|iceland reykjavik|india delhi|indonesia
jakarta|iran tehran|iraq baghdad|ireland dublin|italy rome|jamaica kingston|japan tokyo|
jordan amman|kenya nairobi|latvia riga|lebanon beirut|libya tripoli|lithuania vilnius|
luxembourg luxembourg|malta valletta|mali bamako|morocco rabat|mozambique maputo|nepal
kathmandu|netherlands amsterdam|nicaragua managua|nigeria abuja|norway oslo|peru lima|
philippines manila|poland warsaw|portugal lisbon|romania bucharest|russia moscow|rwanda
kigali|senegal dakar|serbia belgrade|slovakia bratislava|slovenia ljubljana|somalia
mogadishu|spain madrid|sudan khartoum|sweden stockholm|switzerland bern|syria damascus|
tajikistan dushanbe|thailand bangkok|togo lome|tunisia tunis|turkey ankara|uganda kampala|
ukraine kiev|uruguay montevideo|venezuela caracas|vietnam hanoi|zambia lusaka"""

DEMONYM = """france french|spain spanish|germany german|italy italian|japan japanese|
china chinese|russia russian|poland polish|sweden swedish|norway norwegian|denmark danish|
finland finnish|greece greek|turkey turkish|egypt egyptian|india indian|brazil brazilian|
mexico mexican|canada canadian|australia australian|austria austrian|belgium belgian|
bulgaria bulgarian|croatia croatian|cuba cuban|hungary hungarian|iceland icelandic|
ireland irish|israel israeli|korea korean|latvia latvian|lithuania lithuanian|
morocco moroccan|nigeria nigerian|norway norwegian|peru peruvian|portugal portuguese|
romania romanian|serbia serbian|somalia somali|switzerland swiss|thailand thai|
ukraine ukrainian|vietnam vietnamese|scotland scottish|england english|wales welsh|
iran iranian|iraq iraqi|kenya kenyan|argentina argentine|chile chilean|colombia colombian|
ethiopia ethiopian|ghana ghanaian|jamaica jamaican|lebanon lebanese|libya libyan|
malta maltese|nepal nepalese|senegal senegalese|sudan sudanese|syria syrian|tunisia tunisian"""

MALE_FEMALE = """king queen|actor actress|prince princess|uncle aunt|nephew niece|
lion lioness|host hostess|waiter waitress|emperor empress|duke duchess|god goddess|
hero heroine|widower widow|groom bride|monk nun|sir madam|husband wife|father mother|
brother sister|son daughter|boy girl|man woman|male female|bull cow|rooster hen|
stallion mare|ram ewe|drake duck|gander goose|tiger tigress|master mistress|
count countess|baron baroness|lord lady|steward stewardess|poet poetess|
priest priestess|prophet prophetess|abbot abbess|czar czarina|sorcerer sorceress"""

ANIMAL_YOUNG = """cat kitten|dog puppy|cow calf|horse foal|sheep lamb|pig piglet|
goat kid|duck duckling|goose gosling|chicken chick|deer fawn|bear cub|lion cub|
wolf cub|fox cub|kangaroo joey|swan cygnet|eagle eaglet|frog tadpole|butterfly
caterpillar|salmon fry|eel elver|hare leveret|owl owlet|seal pup|whale calf|
elephant calf|rabbit bunny|goat kid|fish fry"""

ANIMAL_SOUND = """dog bark|cat meow|cow moo|lion roar|horse neigh|pig oink|duck quack|
sheep bleat|goat bleat|frog croak|bee buzz|snake hiss|wolf howl|owl hoot|donkey bray|
crow caw|mouse squeak|bird chirp|elephant trumpet|monkey chatter|hen cluck|rooster crow|
turkey gobble|dove coo|lamb bleat|bull bellow|deer bellow"""

ORDINAL = """one first|two second|three third|four fourth|five fifth|six sixth|
seven seventh|eight eighth|nine ninth|ten tenth|eleven eleventh|twelve twelfth|
twenty twentieth|hundred hundredth|thousand thousandth"""

FACTS = {"country_capital": CAPITAL, "country_demonym": DEMONYM,
         "male_female": MALE_FEMALE, "animal_young": ANIMAL_YOUNG,
         "animal_sound": ANIMAL_SOUND, "number_ordinal": ORDINAL}


def morphology(vocab):
    V = set(vocab)
    rels = collections.defaultdict(set)
    for w in vocab:
        if len(w) < 3:
            continue
        for name, fn in SUFFIX.items():
            for cand in fn(w):
                if cand in V and cand != w and len(cand) > len(w):
                    rels[f"cook:suf_{name}"].add((w, cand))
                    break
        for name, fn in PREFIX.items():
            for cand in fn(w):
                if cand in V and cand != w:
                    rels[f"cook:pre_{name}"].add((w, cand))
                    break
    return rels


def facts(vocab):
    V = set(vocab)
    rels = {}
    for name, blob in FACTS.items():
        pairs = set()
        for chunk in blob.replace("\n", "").split("|"):
            parts = chunk.split()
            if len(parts) != 2:
                continue
            a, b = parts
            if a in V and b in V and a != b:
                pairs.add((a, b))
        rels[f"cook:{name}"] = pairs
    return rels


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    m = morphology(vocab)
    f = facts(vocab)
    allr = {**m, **f}
    allr = {k: sorted(v) for k, v in allr.items() if len(v) >= 12}

    print(f"{'cooked relation':<28}{'pairs':>8}{'fan-in':>9}{'distinct targets':>18}")
    print("-" * 65)
    tot = 0
    for r in sorted(allr, key=lambda r: -len(allr[r])):
        tg = collections.Counter(b for _, b in allr[r])
        print(f"{r:<28}{len(allr[r]):>8}{len(allr[r])/len(tg):>9.2f}{len(tg):>18}")
        tot += len(allr[r])
    print(f"\n{len(allr)} cooked relations, {tot} pairs")

    new_targets = {b for pl in allr.values() for _, b in pl}
    print(f"distinct words appearing as a TARGET (i.e. newly derivable): "
          f"{len(new_targets)}")

    wn = json.load(open(f"{D}/relations.json"))
    old_det = set()
    for r, pl in wn.items():
        tg = collections.Counter(b for _, b in pl)
        if len(pl) / max(len(tg), 1) <= 1.2:
            old_det |= set(tg)
    fresh = new_targets - old_det
    V = len(vocab)
    print(f"of those, {len(fresh)} were NOT determinable by any WordNet relation "
          f"with fan-in <= 1.2")
    print(f"\nceiling before cooking : {V/(V-len(old_det)):.3f}x  "
          f"({len(old_det)} determinable)")
    print(f"ceiling after cooking  : {V/(V-len(old_det|new_targets)):.3f}x  "
          f"({len(old_det|new_targets)} determinable)")

    json.dump({k: [list(p) for p in v] for k, v in allr.items()},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
