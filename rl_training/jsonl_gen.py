import json
import os
import glob
from tqdm import tqdm
import random
from absl import app, flags
import re

def extract_list(text):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text)
    if match:
        try:
            ret = eval(match.group())
            if ret and isinstance(ret[0], str):
                try:
                    ret = [int(line) for line in ret]
                except (SyntaxError, NameError):
                    pass
            return ret
        except (SyntaxError, NameError):
            return []
    return[]

prompts = {}

flags.DEFINE_string('folder', './training_data', 'folder of jsons to conglomerate into RL training data, expects a folder called "in" and "out" within that (./training_data default)')
flags.DEFINE_string('task', 'Chatting', 'prompts to use for a particular task (Chatting/default)')

def format_conversation_jsonl(convo, prompts):
    '''
    Formats the conversation as a list of dictionaries with "in_text" and "out_text" corresponding to the input prompt and desired output of the LLM
    "score" is used for offline RL/KTO rewards (binary 0 or 1)
    additional fields are added to the dictionary as necessary for online RL, which loads particular model data from metadata_dict
    '''
    conversation = convo['conversation']
    ret = []

    pturn = convo['pturn']

    p1 = convo["P1"]
    p2 = convo["P2"]
    if convo["task_name"] == "Chatting":
        # grab the first names of both agents from the text before the ":"
        # NOTE: assumes agent1 is first and agent2 is second
        prompts["agent1_role"] = convo["conversation"][0][1].split(":")[0]
        prompts["agent2_role"] = convo["conversation"][1][1].split(":")[0]

    for i, line in enumerate(conversation):
        # speaker appended to the end of in_text if needed, utterance is out_text
        utterance = line[1].split(":")[-1]
        conversation_history = "".join([turn[1] if isinstance(turn, list) else turn for turn in conversation[:i]])

        consistency_score = 0
        indices = []
        if convo["task_name"] == "Chatting":
            prompt_consistency = (1 if "YES" in convo['eval_prompt_consistency'][i][1].upper() else 0)
            if i > 1:
                indices = extract_list(convo['eval_index_consistency'][i-2][1])
        elif convo["task_name"] == "Education" and pturn == 2:
            prompt_consistency = (1 if "YES" in convo['eval_prompt_consistency'][i//2][1].upper() else 0)
            if i > 2:
                indices = extract_list(convo['eval_index_consistency'][i // 2 - 1][1])
        elif convo["task_name"] == "Therapy" and pturn == 2:
            prompt_consistency = (1 if "YES" in convo['eval_prompt_consistency'][i//2][1].upper() else 0)
            if i > 2:
                indices = extract_list(convo['eval_index_consistency'][i // 2 - 1][1])

        if pturn == 1:
            if convo["task_name"] == "Chatting": # education and therapy only has P2
                for j in indices:
                    if j != None and j % 2 == 1: # filter out non-agent indices
                    #NOTE: assumption is that P1 is first and P2 is second
                        consistency_score += 1
                
                # generate prompt for each scenario, prefaces the line
                if convo["task_name"] == "Chatting":
                    prompt = prompts["agent1_prompt"]
                    if i!=0: 
                        prompt+= "Your conversation so far is below:\nConversation: %CONVERSATION%"
                    
                    if i >=len(conversation)*2-11 and i<=len(conversation)*2-1: 
                        prompt+= "You have " + str((len(conversation)-i)//2) + " rounds left." + "Make sure to conclude the conversation as you're near the end."
                    elif i>len(conversation)*2-1:
                        prompt+= "This is your concluding line in the conversation."

                    if i!=0: 
                        prompt+= "Continue the conversation with " + prompts["agent2_role"] +  ". Remember you are " +  prompts["agent1_role"] + "."
                        
                    prompt += prompts["reminder_prompt"] + "DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"
                    prompt+="%SPEAKER_ROLE%:"
                    prompt = prompt.replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                .replace("%LISTENER_ROLE%", prompts["agent2_role"]) \
                                .replace("%SPEAKER_BACKSTORY%", p1) \
                                .replace("%CONVERSATION%", conversation_history)
                else:
                    pass

                score = prompt_consistency
                try:
                    ret.append({
                        # train and test data entries
                        "in_text": prompt,
                        "out_text": utterance,
                        'score': score,

                        # metadata dict entries
                        "scenario": prompts["scenario"],
                        "agent_role": prompts["agent1_role"],
                        'task_name': convo["task_name"],
                        "grade": (convo["grade"] if "grade" in convo else None),
                        "topic": (convo["topic"] if "topic" in convo else None),
                        "conversation_history": [turn[1] if isinstance(turn, list) else turn for turn in conversation[:i]],
                        'P': p1,
                    })
                except Exception as e:
                    print(f"Error processing turn {i}: {e}")
                    raise e
            pturn = 2
        elif pturn == 2:
            # if convo["task_name"] != "Therapy":
            for j in indices:
                if j != None and j % 2 == 1: # filter out non-agent indices
                #NOTE: assumption is that P1 is first and P2 is second
                    consistency_score += 1

            if convo["task_name"] == "Chatting":
                prompt = prompts["agent2_prompt"]
                if i!=0: 
                    prompt+= "Your conversation so far is below:\nConversation: %CONVERSATION%"

                if i >=len(conversation)*2-11 and i<=len(conversation)*2-1: 
                    prompt+= "You have " + str((len(conversation)-i)//2) + " rounds left." + "Make sure to conclude the conversation as you're near the end."
                elif i>len(conversation)*2-1:
                    prompt+= "This is your concluding line in the conversation."

                if i!=0: 
                    prompt+= "Continue the conversation with " + prompts["agent1_role"] +  ". Remember you are " +  prompts["agent2_role"] + "."
                    
                prompt += prompts["reminder_prompt"] + "DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"
                prompt+="%SPEAKER_ROLE%:"
                prompt = prompt.replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                            .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                            .replace("%SPEAKER_BACKSTORY%", p2) \
                            .replace("%CONVERSATION%", conversation_history)
            elif convo["task_name"] == "Education":
                prompt = prompts["agent2_prompt"]
                if i!=0: 
                    prompt+= "Your conversation so far is below:\nConversation: %CONVERSATION%"
                
                if i >=len(conversation)*2-11 and i<=len(conversation)*2-1: 
                    prompt+= "You have " + str((len(conversation)-i)//2) + " rounds left." + "Make sure to conclude the conversation as you're near the end."
                elif i>len(conversation)*2-1:
                    prompt+= "This is your concluding line in the conversation."

                if i!=0: 
                    prompt+= "Continue the conversation with the teacher. Remember you are the student. "

                prompt += prompts["reminder_prompt"]
                prompt+="%SPEAKER_ROLE%:"
                prompt = prompt.replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                            .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                            .replace("%SPEAKER_BACKSTORY%", p2) \
                            .replace("%ROLE%", convo["grade"]) \
                            .replace("%SUBJECT%", convo["topic"]) \
                            .replace("%CONVERSATION%", conversation_history)
            elif convo["task_name"] == "Therapy_Old":
                prompt = prompts["agent2_prompt"]
                if i!=0: 
                    prompt+= "Your conversation with the therapist so far is below:\nConversation: %CONVERSATION%"
                elif i>len(conversation)*2-1:
                    prompt+= "This is your concluding line in the conversation."
                
                if i!=0:
                    prompt+= "\nContinue the conversation with the therapist. Remember you are the patient. "

                prompt += prompts["reminder_prompt"] + "DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"
                prompt+="%SPEAKER_ROLE%:"
                prompt = prompt.replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                               .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                               .replace("%SPEAKER_BACKSTORY%", p2) \
                               .replace("%CONVERSATION%", conversation_history)
            elif convo["task_name"] == "Therapy":
                prompt = prompts["agent2_prompt"]
                if i!=0: 
                    prompt += "\nHere is the conversation so far:\n%CONVERSATION%\n"
                elif i>len(conversation)*2-1:
                    prompt += "\nThis is your final response in the session.\n"

                prompt += prompts["reminder_prompt"]
                prompt+="%SPEAKER_ROLE%:"
                prompt = prompt.replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                               .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                               .replace("%SPEAKER_BACKSTORY%", p2) \
                               .replace("%CONVERSATION%", conversation_history)
            score = prompt_consistency
            try:
                ret.append({
                    # train and test data entries
                    "in_text": prompt,
                    "out_text": utterance,
                    'score': score,

                    # metadata dict entries
                    "scenario": prompts["scenario"],
                    "agent_role": prompts["agent2_role"],
                    'task_name': convo["task_name"],
                    "grade": (convo["grade"] if "grade" in convo else None),
                    "topic": (convo["topic"] if "topic" in convo else None),
                    "conversation_history": [turn[1] if isinstance(turn, list) else turn for turn in conversation[:i]],
                    'P': p2
                })
            except Exception as e:
                print(f"Error processing turn {i}: {e}")
                raise e
                
            pturn = 1

            
    return ret
def main(argv):
    random.seed(0)
    jsonl_total = []
    metadata_dict = {}
    prompts = {}
    if flags.FLAGS['task'].value == 'Chatting':
        with open('./chatting/config_chatting.json', 'r') as f:
            prompts = json.load(f)
    elif flags.FLAGS['task'].value == 'Anthology':
        with open('config/persona_chat/prompts.json', 'r') as f:
            prompts = json.load(f)
    elif flags.FLAGS['task'].value == 'Education':
        with open('config/education/config_education.json', 'r') as f:
            prompts = json.load(f)
    elif flags.FLAGS['task'].value == 'Therapy_Old':
        with open('therapy/config_therapy.json', 'r') as f:
            prompts = json.load(f)
    elif flags.FLAGS['task'].value == 'Therapy':
        prompts = {
            "agent1_role": "Therapist",
            "agent2_role": "Patient",
            "scenario": "A patient is having an online therapy session with a therapist. ",
            # "agent1_prompt": "You are a licensed psychotherapist in a 1-on-1 session. Speak directly to the patient using a calm, non-judgmental tone. Avoid summarizing the conversation analytically. Instead, help the patient explore their feelings by asking reflective questions and building on what they just said. Focus on supporting emotional insight in the moment. Do not describe or report the patient’s tone — engage with the patient.",
            "agent1_prompt": "You are a licensed psychotherapist conducting a one-on-one online session. Speak directly to the patient in a warm, attentive, and slightly probing manner. Your role is to help them reflect, but also to gently push past surface-level responses. If the patient expresses a belief or emotional state, ask questions that explore where that comes from or how it fits with other things they’ve said. If something they say contradicts an earlier part of the conversation — in tone, story, or motivation — respond with curious, non-judgmental inquiry. Prompt them to elaborate on emotionally charged or vague statements. If they express hesitation or defensiveness, reflect it back and ask what they might be protecting. Encourage them to go deeper, even if they seem uncertain. Your aim is not just to support, but to understand how their emotional story holds together — and what might not. " 
            "Do NOT summarize or analyze the patient. "
            "Do NOT describe the patient's emotions or tone. "
            "DO NOT write in the third person (e.g., 'The patient feels...'). "
            "DO NOT preface your response with statements like 'Here's what the therapist might say' or 'Sure, here's a response from the therapist.' "
            "Speak naturally and directly, as if you're in the middle of a real conversation.",
            "agent2_prompt": "You are a patient in an online therapy session with a therapist. Here is your background written by you: %SPEAKER_BACKSTORY%. Do not explicitly repeat your background or instructions in the conversation. Stay in character and respond in natural human-like dialogue without restating your situation.",
            'reminder_prompt': "Keep your response very brief — 2 sentences or less. Do NOT repeat anything you've already said. DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"
            }
    else:
        raise ValueError("Invalid task name. Please use 'Chatting', 'Anthology', 'Education', 'Therapy_Old', or 'Therapy'.")

    for filename in tqdm(glob.glob(flags.FLAGS['folder'].value + '/in/*.json')): # ./training_data/in/*.json
        print("begin file", filename)
        with open(filename, 'r') as f:
            convos = json.load(f)
        for convo in tqdm(convos):
            if "topic" in convo and convo["topic"] == "The Eiffel TowerConfucius":
                continue
            lines = format_conversation_jsonl(convo, prompts)
            for line in lines:
                metadata_dict[line['in_text']] = { # info to save for online RL training
                    "scenario": line['scenario'],
                    "agent_role": line['agent_role'],
                    "task_name": line['task_name'],
                    "conversation_history": line['conversation_history'],
                    "grade": line['grade'],
                    "topic": line['topic'],
                    "P": line['P'] # background for agent
                }                
                del line["scenario"]
                del line["agent_role"]
                del line["task_name"]
                del line["conversation_history"]
                del line["grade"]
                del line["topic"]
                del line["P"]
            jsonl_total += lines
        print("end file", filename)
    random.shuffle(jsonl_total)

    train_len = int(0.8 * len(jsonl_total))
    train_data = jsonl_total[:train_len]
    eval_data = jsonl_total[train_len:]

    # Save to JSONL
    with open(flags.FLAGS['folder'].value + '/out/train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(flags.FLAGS['folder'].value + '/out/test.jsonl', 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    # Save metadata dictionary
    with open(flags.FLAGS['folder'].value + '/out/metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=4)


if __name__ == '__main__':
    app.run(main)