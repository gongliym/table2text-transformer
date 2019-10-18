import sys
import codecs
import argparse
import json
from datetime import datetime
from collections import Counter, defaultdict
from text2num import text2num, NumberException
from tokenizer import word_tokenize, sent_tokenize

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

def _get_player_index(game):
    home_players, vis_players = [], []
    nplayers = len(game["box_score"]["PTS"].keys())
    if game["home_city"] != game["vis_city"]:
        for index in [str(x) for x in range(nplayers)]:
            player_city = game["box_score"]["TEAM_CITY"][index]
            if player_city == game["home_city"]:
                if len(home_players) < NUM_PLAYERS:
                    home_players.append(index)
            else:
                if len(vis_players) < NUM_PLAYERS:
                    vis_players.append(index)
    else:
        for index in range(nplayers):
            if index < nplayers/2:
                home_players.append(str(index))
            else:
                vis_players.append(str(index))
    return home_players, vis_players

def _get_win_loss(game):
    home_score = int(game['home_line']['TEAM-PTS'])
    vis_score  = int(game['vis_line']['TEAM-PTS'])
    if home_score >= vis_score: # normally there is no tie game
        return ('W', 'L')
    elif vis_score > home_score:
        return ('L', 'W')

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

def extract_player_stats(game, player_key, player_feat):
    result = []
    for key in bs_keys:
        rel_entity = game["box_score"][player_name_key][player_key] if player_key is not None else "N/A"
        rel_type = key.split('-')[1]
        rel_value = game["box_score"][rel_type][player_key] if player_key is not None else "N/A"
        rel_item = '{}|{}|{}|{}'.format(rel_entity, rel_type, rel_value, player_feat)
        result.append(rel_item)
    return result

def player_stats_number():
    return len(bs_keys)

def extract_team_stats(game):
    home_result = []
    vis_result = []
    home_winloss, vis_winloss = _get_win_loss(game)
    for key in ls_keys:
        home_entity= game["home_line"]['TEAM-NAME']
        home_value = game["home_line"][key]
        home_type  = key
        home_feat  = 'H/{}'.format(home_winloss)

        vis_entity = game["vis_line"]['TEAM-NAME']
        vis_value  = game["vis_line"][key]
        vis_type   = key
        vis_feat   = 'V/{}'.format(vis_winloss)

        home_item = '{}|{}|{}|{}'.format(home_entity, home_type, home_value, home_feat)
        vis_item  = '{}|{}|{}|{}'.format(vis_entity, vis_type, vis_value, vis_feat)

        home_result.append(home_item)
        vis_result.append(vis_item)
    return home_result, vis_result

def extract_date_info(game):
    weekday = datetime.strptime(game['day'], '%m_%d_%y').strftime('%A')
    return 'N/A|N/A|{}|N/A'.format(weekday)

def team_stats_number():
    return len(ls_keys)

def extract_tables(json_data):
    gametable = []
    for game in json_data:
        game_stats = []
        home_players, vis_players = _get_player_index(game)
        home_winloss, vis_winloss = _get_win_loss(game)
        
        # Add home player stats
        for idx in range(NUM_PLAYERS):
            player_key = home_players[idx] if idx < len(home_players) else None
            player_feat = 'H/{}'.format(home_winloss)
            player_stats = extract_player_stats(game, player_key, player_feat)
            game_stats.extend(player_stats)
            
        # Add vis player stats
        for idx in range(NUM_PLAYERS):
            player_key = vis_players[idx] if idx < len(vis_players) else None
            player_feat = 'V/{}'.format(vis_winloss)
            player_stats = extract_player_stats(game, player_key, player_feat)
            game_stats.extend(player_stats)

        # process team
        home_result, vis_result = extract_team_stats(game)
        assert len(home_result) == len(vis_result)
        game_stats.extend(home_result)
        game_stats.extend(vis_result)

        # weekday
        weekday = extract_date_info(game)
        game_stats.append(weekday)

        game_stats = ['_'.join(each.split()) for each in game_stats]
        
        assert len(gametable) == 0 or len(game_stats) == len(gametable[-1])
        gametable.append(game_stats)
    return gametable

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

def get_links(game, summary_entities, summary_numbers):
    links = []
    bs_scores = game['box_score']
    home_line_score = game['home_line']
    vis_line_score = game['vis_line']
    for number_item in summary_numbers:
        start_idx, end_idx, number_string, number_value = number_item
        for entity_item in summary_entities:
            (entity_start_idx, entity_end_idx, entity_name, entity_type) = entity_item
            if entity_type.startswith('home') or entity_type.startswith('vis'):
                if entity_type.startswith('home'):
                    linescore = home_line_score
                elif entity_type.startswith('vis'):
                    linescore = vis_line_score
                else:
                    assert False, "entity type wrong value: {}".format(entity_type)
                
                for ls_key in ls_keys:
                    if linescore[ls_key] == number_string \
                        or linescore[ls_key] == str(number_value):
                        links.append((start_idx, end_idx, entity_start_idx, entity_end_idx, ls_key))

            elif entity_type == 'TEAM-CITY': # Los Angeles
                for ls_key in ls_keys:
                    if home_line_score[ls_key] == number_string \
                        or home_line_score[ls_key] == str(number_value):
                        links.append((start_idx, end_idx, entity_start_idx, entity_end_idx, ls_key))
                    if vis_line_score[ls_key] == number_string \
                        or vis_line_score[ls_key] == str(number_value):
                        links.append((start_idx, end_idx, entity_start_idx, entity_end_idx, ls_key))
            else:
                player_key = entity_type
                for bs_key in bs_keys:
                    bs_key = bs_key[7:] # remove PLAYER- prefix
                    if bs_scores[bs_key][player_key] == number_string \
                        or bs_scores[bs_key][player_key] == str(number_value):
                        links.append((start_idx, end_idx, entity_start_idx, entity_end_idx, bs_key))

    return links
    
def extract_links(json_data, summary_list, entity_list, verbose=False):
    assert len(json_data) == len(summary_list)
    assert len(json_data) == len(entity_list)
    link_list = []
    for game, summary, entities in zip(json_data, summary_list, entity_list):
        summary_text = ' '.join(summary)
        summary_sents = sent_tokenize(summary_text)
        if verbose:
            print(len(link_list), "Entities from Json:", entities)

        links = {}
        sent_start_token_index = 0
        for sent in summary_sents:
            sent = sent.strip()
            sent_words = sent.split()
            summary_entities = extract_summary_entities(sent_words, entities)
            summary_numbers  = extract_summary_numbers(sent_words)
            sent_links = get_links(game, summary_entities, summary_numbers)
            for each in sent_links:
                value_start_idx, _, entity_start_idx, _, type_key = each
                example_key = (sent, entity_start_idx, value_start_idx)
                if example_key not in links:
                    links[example_key] = type_key
                else:
                    if type_key.startswith('TEAM'):
                        if type_key in links.keys():
                            continue
                        if type_key == 'TEAM-WINS':
                            links[example_key] = type_key
                        elif type_key == 'TEAM-LOSSES':
                            links[example_key] = type_key

                    if check_sth(sent, entity_start_idx, value_start_idx, type_key):
                        links[example_key] = type_key

        link_list.append(links)

    return link_list

def check_sth(sent, entity_start_idx, value_start_idx, type_key):
    sent_words = sent.split()
    if len(sent_words) <= value_start_idx + 1:
        return False
    info_token = sent_words[value_start_idx+1]

    if info_token.startswith('assist') and type_key == 'AST':
        return True
    elif info_token.startswith('rebound') and type_key == 'REB':
        return True
    elif info_token.startswith('point') and type_key == 'PTS':
        return True
    elif info_token.startswith('steal') and type_key == 'STL':
        return True
    elif info_token.startswith('block') and type_key == 'BLK':
        return True
    elif info_token.startswith('min') and type_key == 'MIN':
        return True
    elif info_token.startswith('turnover') and type_key == 'TO':
        return True
    else:
        return False

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
    # convert json to table
    table_list = extract_tables(json_data)
    # get summary
    entity_dict = extract_entities(json_data)
    summary_list = extract_summary(json_data, summary_key, entity_dict)
    assert len(table_list) == len(summary_list)

    game_entity_list = extract_game_entities(json_data)
    assert len(table_list) == len(game_entity_list)
    links = extract_links(json_data, summary_list, game_entity_list, args.verbose)

    for game in links:
        for example in game:
            (sent, entity_start_idx, value_start_idx) = example
            target = game[example]
            print("{}\t{}\t{}\t{}".format(sent, entity_start_idx, value_start_idx, target))
