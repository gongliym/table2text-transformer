import sys
import codecs
import argparse
import json
from datetime import datetime
from collections import Counter, defaultdict
from text2num import text2num, NumberException
from tokenizer import word_tokenize, sent_tokenize

NOREL = 'NOREL'
NUM_PLAYERS = 13
player_name_key="PLAYER_NAME"
bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
     "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
     "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
     "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
     "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
    "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
    "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])

def extract_summary_entities(words, entity_dict):
    entities = []
    idx = 0
    while idx < len(words):
        if words[idx] in entity_dict:
            entity_name = words[idx]
            entity_types = entity_dict[entity_name]
            if isinstance(entity_types, set):
                for each_type in entity_types:
                    entities.append((idx, idx+1, entity_name, each_type))
            else:
                entities.append((idx, idx+1, entity_name, entity_types))
        idx += 1
    return entities

def extract_summary_numbers(words):
    ignores = set(["three point", "three - point", "three - pt", "three pt", "three - pointers", "three pointers", "three pointer"])
    numbers = []
    idx = 0
    while idx < len(words):
        is_number = False
        try:
            number_value = int(words[idx])
            numbers.append((idx, idx+1, words[idx], number_value))
            idx += 1
            continue
        except:
            pass
        for end_idx in range(min(idx+5, len(words)), idx, -1):
            number_string = ' '.join(words[idx:end_idx])
            try:
                number_value = text2num(number_string)
                numbers.append((idx, end_idx, number_string, number_value))
                is_number = True
                idx = end_idx
                break
            except NumberException:
                if number_string in ignores:
                    break
        if not is_number:
            idx += 1
    return numbers


def extract_entities(json_data):
    entities = {}
    for game in json_data:
        entities[game['home_name']] = 'TEAM-NAME'
        entities[game["home_line"]["TEAM-NAME"]] = 'TEAM-NAME'
        entities[game['vis_name']] = 'TEAM-NAME'
        entities[game["vis_line"]["TEAM-NAME"]] = 'TEAM-NAME'

        entities[game['home_city']] = 'TEAM-CITY'
        entities[game['vis_city']] = 'TEAM-CITY'
        entities['LA'] = 'TEAM-CITY'
        entities['Los Angeles'] = 'TEAM-CITY'

        for player_key in game['box_score']['PLAYER_NAME']:
            player_name = game['box_score']['PLAYER_NAME'][player_key]
            player_first = game['box_score']['FIRST_NAME'][player_key]
            player_second = game['box_score']['SECOND_NAME'][player_key]
            entities[player_name] = 'PLAYER-NAME'
            entities[player_first] = 'PLAYER-NAME'
            entities[player_second] = 'PLAYER-NAME'
        
        for entity in list(entities.keys()):
            tokens = entity.strip().split()
            if len(tokens) == 1: continue
            for part in tokens:
                if len(part) > 1 and part not in ["II", "III", "Jr.", "Jr"]:
                    if part not in entities:
                        entities[part] = entities[entity]
                    elif isinstance(entities[part], set):
                        entities[part].add(entities[entity])
                    elif entities[part] != entities[entity]:
                        a_set = set()
                        a_set.add(entities[entity])
                        a_set.add(entities[part])
                        entities[part] = a_set
    return entities

def extract_summary(json_data, summary_key, entity_dict):
    summary_list = []

    for game in json_data:
        summary = game.get(summary_key, None)
        assert summary is not None
        words = ' '.join(summary).strip().split()
        result = []
        idx = 0
        while idx < len(words):
            if words[idx] in entity_dict:
                length = 1
                while idx + length <= len(words) and ' '.join(words[idx:idx+length]) in entity_dict:
                    length += 1
                length -= 1
                result.append('_'.join(words[idx:idx+length]))
                idx += length
            else:
                result.append(words[idx])
                idx += 1

        # result_tokens = word_tokenize(' '.join(result), language='english')
        summary_list.append(result)
    return summary_list

def extract_links(json_data, summary_list, entity_list, verbose=False):
    assert len(json_data) == len(summary_list)
    assert len(json_data) == len(entity_list)
    link_list = []
    for game, summary, entities in zip(json_data, summary_list, entity_list):
        summary_text = ' '.join(summary)
        summary_sents = sent_tokenize(summary_text)

        links = {}
        sent_start_token_index = 0
        for sent in summary_sents:
            sent = sent.strip()
            sent_words = sent.split()
            summary_entities = extract_summary_entities(sent_words, entities)
            summary_numbers  = extract_summary_numbers(sent_words)
            sent_links = get_links(game, sent, summary_entities, summary_numbers)
            for each in sent_links:
                entity_start_idx, value_start_idx, type_key = each
                example_key = (sent, entity_start_idx, value_start_idx)
                if example_key not in links:
                    links[example_key] = type_key
                else:
                    if type_key.startswith('TEAM'):
                        if type_key == 'TEAM-WINS':
                            links[example_key] = type_key
                        elif type_key == 'TEAM-LOSSES':
                            links[example_key] = type_key

                    if check_sth(sent, entity_start_idx, value_start_idx, type_key):
                        links[example_key] = type_key

        link_list.append(links)

    return link_list


def get_links(game, sent, summary_entities, summary_numbers):
    links = []
    bs_scores = game['box_score']
    home_line_score = game['home_line']
    vis_line_score = game['vis_line']
    for number_item in summary_numbers:
        number_start_idx, _, number_string, number_value = number_item
        for entity_item in summary_entities:
            entity_start_idx, _, entity_name, entity_type = entity_item
            if entity_type.startswith('home') or entity_type.startswith('vis'):
                if entity_type.startswith('home'):
                    linescore = home_line_score
                elif entity_type.startswith('vis'):
                    linescore = vis_line_score
                else:
                    assert False, "entity type wrong value: {}".format(entity_type)
                found = False
                for ls_key in ls_keys:
                    if linescore[ls_key] == number_string \
                        or linescore[ls_key] == str(number_value):
                        links.append((entity_start_idx, number_start_idx, ls_key))
                        found = True
                if not found:
                    links.append((entity_start_idx, number_start_idx, NOREL))
            elif entity_type == 'TEAM-CITY': # Los Angeles
                found = False
                for ls_key in ls_keys:
                    if home_line_score[ls_key] == number_string \
                        or home_line_score[ls_key] == str(number_value):
                        links.append((entity_start_idx, number_start_idx, ls_key))
                        found = True
                    if vis_line_score[ls_key] == number_string \
                        or vis_line_score[ls_key] == str(number_value):
                        links.append((entity_start_idx, number_start_idx, ls_key))
                        found = True
                if not found:
                    links.append((entity_start_idx, number_start_idx, NOREL))
            else:
                player_key = entity_type
                found = False
                for bs_key in bs_keys:
                    bs_key = bs_key[7:] # remove PLAYER- prefix
                    if bs_scores[bs_key][player_key] == number_string \
                        or bs_scores[bs_key][player_key] == str(number_value):
                        if check_sth(sent, entity_start_idx, number_start_idx, bs_key):
                            links.append((entity_start_idx, number_start_idx, bs_key))
                            found = True
                if not found:
                    links.append((entity_start_idx, number_start_idx, NOREL))
    return links


def check_sth(sent, entity_start_idx, value_start_idx, type_key):
    sent_words = sent.split()
    if len(sent_words) <= value_start_idx + 1:
        return False
    info_token = sent_words[value_start_idx+1]

    if info_token.startswith('assist') and type_key != 'AST':
        return False
    elif info_token.startswith('rebound') and type_key != 'REB':
        return False
    elif info_token.startswith('point') and type_key != 'PTS':
        return False
    elif info_token.startswith('steal') and type_key != 'STL':
        return False
    elif info_token.startswith('block') and type_key != 'BLK':
        return False
    elif info_token.startswith('min') and type_key != 'MIN':
        return False
    elif info_token.startswith('turnover') and type_key != 'TO':
        return False
    else:
        return True

def extract_game_entities(json_data):
    entity_list = []
    for game in json_data:
        entities = {}

        entities[game['home_name']] = 'home TEAM-NAME'
        entities[game["home_line"]["TEAM-NAME"]] = 'home TEAM-NAME'
        entities[game['vis_name']] = 'vis TEAM-NAME'
        entities[game["vis_line"]["TEAM-NAME"]] = 'vis TEAM-NAME'

        entities[game['home_city']] = 'home TEAM-CITY'
        entities[game['vis_city']] = 'vis TEAM-CITY'

        if game["home_city"] == "Los Angeles" or game["home_city"] == 'LA':
            entities['LA'] = 'home TEAM-CITY'
            entities['Los Angeles'] = 'home TEAM-CITY'

        if game["vis_city"] == "Los Angeles" or game["vis_city"] == 'LA':
            entities['LA'] = 'vis TEAM-CITY'
            entities['Los Angeles'] = 'vis TEAM-CITY'

        if game["home_city"] == game["vis_city"]:
            entities['LA'] = 'TEAM-CITY'
            entities['Los Angeles'] = 'TEAM-CITY'

        for player_key in game['box_score']['PLAYER_NAME']:
            player_name = game['box_score']['PLAYER_NAME'][player_key]
            player_first = game['box_score']['FIRST_NAME'][player_key]
            player_second = game['box_score']['SECOND_NAME'][player_key]
            entities[player_name] = player_key
            entities[player_first] = player_key
            entities[player_second] = player_key

        for name in game['box_score']['TEAM_CITY'].values():
            assert name in entities

        for entity in list(entities.keys()):
            parts = entity.strip().split()
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 1 and part not in ["II", "III", "Jr.", "Jr"]:
                        if part not in entities:
                            entities[part] = entities[entity]
                        elif isinstance(entities[part], set):
                            entities[part].add(entities[entity])
                        elif entities[part] != entities[entity]:
                            a_set = set()
                            a_set.add(entities[entity])
                            a_set.add(entities[part])
                            entities[part] = a_set
        
        result = {}
        for each in entities:
            key = '_'.join(each.split())
            result[key] = entities[each]

        entity_list.append(result)
    return entity_list

if __name__ == '__main__':
    readme = """
    """
    parser = argparse.ArgumentParser(description=readme)
    parser.add_argument("-d", "--data",   required=True, help = "rotowire json data")
    parser.add_argument("-o", "--output", required=True, help = "output prefix")
    parser.add_argument('-v', "--verbose", action='store_true', help = "verbose")
    args = parser.parse_args()

    json_data = json.load(open(args.data, 'r'))
    summary_key = 'summary'

    entity_dict = extract_entities(json_data)
    summary_list = extract_summary(json_data, summary_key, entity_dict)
    game_entity_list = extract_game_entities(json_data)
    links = extract_links(json_data, summary_list, game_entity_list, args.verbose)

    data_outf = open(args.output+".examples", 'w')
    for game in links:
        for example in game:
            (sent, entity_start_idx, value_start_idx) = example
            target = game[example]
            data_outf.write("{}\t{}\t{}\t{}\n".format(sent, entity_start_idx, value_start_idx, target))
    data_outf.close()

    dict_outf = open(args.output+".entities", 'w')
    entity_count = {}
    for game in game_entity_list:
        for entity in game:
            if entity not in entity_count:
                entity_count[entity] = 1
            else:
                entity_count[entity] += 1
    for entity, cnt in sorted(entity_count.items(), key=lambda x:x[1]):
        dict_outf.write(entity+"\n")
    dict_outf.close()
            
