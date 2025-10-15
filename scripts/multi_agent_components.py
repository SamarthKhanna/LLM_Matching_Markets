import pandas as pd
import os

from groq import Groq
from openai import OpenAI
import anthropic
from google import genai
import os
from pydantic import BaseModel, ConfigDict
from enum import Enum

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
# from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
# from semantic_kernel import Kernel
# from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
# from semantic_kernel.memory.null_memory import NullMemory


class LLM():
    def __init__(self, auth_file = 'Auth_keys/gemini_auth_key_2.txt', family='gemini', model='gemini-1.0-pro-latest') -> None:
        self.auth_file = auth_file
        self.family = family
        self.model = model
        self.tokens = 0
        if self.family == 'gemini':
            self.init_gemini()
        if self.family == 'mixtral':
            self.init_mixtral()
        if self.family == 'llama':
            self.init_llama()
        if self.family == "chatgpt":
            self.init_gpt()
        # if self.family == 'claude':
        #     self.init_claude()
        if self.family == 'deepseek':
            self.init_deepseek()

    def init_gemini(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        # genai.configure(api_key=AUTHKEY)

        # self.llm = genai.GenerativeModel(self.model)
        # self.chat = self.llm.start_chat(history=[])

        self.llm = genai.Client(api_key=AUTHKEY)
        self.autogen_client = OpenAIChatCompletionClient(
            model=self.model,
            family=self.model,
            # model="o3-mini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=AUTHKEY, # Optional if you have an OPENAI_API_KEY env variable set.
        )
            
        # client = AsyncOpenAI(
        #     api_key=AUTHKEY,
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        # )
        # self.llm = client.chat.completions

        # config = OpenAIConfig(self.model)
        # self.llm = models.openai(client, config)


    def init_llama(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        os.environ["GROQ_API_KEY"]=AUTHKEY

        self.llm = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.autogen_client = OpenAIChatCompletionClient(
            model=self.model,
            # model="o3-mini",
            base_url="https://api.groq.com/openai/v1",
            family='unknown',
            api_key=os.environ.get("GROQ_API_KEY"), # Optional if you have an OPENAI_API_KEY env variable set.
        )

    def init_gpt(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()
        os.environ["OPENAI_API_KEY"] = AUTHKEY

        client = OpenAI()
        self.llm = client.chat.completions
        self.autogen_client = OpenAIChatCompletionClient(
            model=self.model,
            # model="o3-mini",
            api_key=AUTHKEY, # Optional if you have an OPENAI_API_KEY env variable set.
        )

    def init_deepseek(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()
        os.environ["OPENAI_API_KEY"] = AUTHKEY

        client = OpenAI(base_url="https://api.deepseek.com")
        self.llm = client.chat.completions
        self.autogen_client = OpenAIChatCompletionClient(
            model=self.model,
            # model="o3-mini",
            base_url="https://api.deepseek.com",
            family='r1',
            api_key=AUTHKEY, # Optional if you have an OPENAI_API_KEY env variable set.
        )

    # def init_claude(self):
    #     with open(self.auth_file, 'r') as f:
    #         AUTHKEY = f.read()

    #     self.llm = anthropic.Anthropic(api_key=AUTHKEY)
    #     sk_client = AnthropicChatCompletion(
    #         ai_model_id=self.model,
    #         api_key=AUTHKEY,
    #         # service_id="my-service-id",  # Optional; for targeting specific services within Semantic Kernel
    #     )
    #     # settings = AnthropicChatPromptExecutionSettings(
    #     #     temperature=0,
    #     # )

    #     self.autogen_client = SKChatCompletionAdapter(
    #         sk_client, kernel=Kernel(memory=NullMemory()),
    #     )

    def makeLLMRequestFormatting(self, queryText, json_schema):
        print(json_schema)
        response =  self.llm.models.generate_content(
            model = self.model,
            contents = queryText,
            # config={
            #     'resonse_mime_type': 'application/json',
            #     'response_schema': json_schema
            # }
        )
        response = response.text
        return response
    
    def makeLLMRequest(self, queryText):
        usage = None
        if self.family == "gemini":
            # response = self.chat.send_message(
            #     content=queryText,
            #     safety_settings={
            #         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            #         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            #     },
            #     # config={
            #     #     'resonse_mime_type': 'application/json',
            #     #     'response_schema': json_schema
            #     # }
            #     # generation_config=genai.types.GenerationConfig(
            #     #     # Only one candidate for now.
            #     #     # candidate_count=1,
            #     #     # stop_sequences=["x"],
            #     #     # max_output_tokens=20,
            #     #     temperature=0.0,
            #     #     resonse_mime_type = 'application/json',
            #     #     response_schema = json_schema
            #     # ),
            # )
            # response = response.text
            response =  self.llm.models.generate_content(
                model = self.model,
                contents = queryText,
            )
            usage = response.usage_metadata
            response = response.text
            # generator = generate.json(self.llm, Matching10)
            # response = generator(queryText)
            # print(response)

        elif self.family == 'llama':
            response = self.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": queryText,
                    }
                ],
                model=self.model,
                # temperature=0
            )
            usage = response.usage
            response = response.choices[0].message.content

        elif self.family in ("chatgpt", "deepseek"):
            messages = [
                {"role": "user", "content": queryText},
            ]
            response = self.llm.create(messages=messages, model=self.model)
            self.tokens += response.usage.total_tokens
            usage = response.usage
            response = response.choices[0].message.content

        elif self.family == "claude":
            response = self.llm.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    # temperature=0,
                    messages=[
                        {"role": "user", "content": queryText}
                ]
            )
            response = response.content[0].text

        return response, usage
    
    def get_tokens_used(self):
        return self.tokens


class MultiAgentRoundRobinPt5():
    def __init__(self, instance_file, model_client) -> None:
        self.size = int(instance_file.split('_')[0][1:])
        self.instance_file = f"../instances_matchings/{instance_file}"
        self.rrgc_list = []
        self.instances = pd.read_csv(self.instance_file)
        self.client = model_client

    
# Define a tool using a Python function.
    def preference_analysis(womanPref: str, proposingMan: str, woman: str, womanPartner: str) -> str:
        """
        Use this function to determine if a matched woman (ex. W1) should accept a proposal from a man (ex. M1)
        If a woman has an existing partner and another man proposes to her, call this function. Use the following argments:
        womanPref should be a string of the form "[M1,M2,M3,M4,M5]"
        proposingMan should be a man of the form "M1"
        woman should be a woman of the form "W1". This is the woman who is being proposed to
        womanPartner should be a man of the form "M1". This is the woman's current partner
        """
        strippedPref = womanPref.replace(" ", "").strip("[]")  # Remove the leading and trailing square brackets
        prefList = strippedPref.split(",")  # Split the string by the comma delimiter
        # print(prefList)

        proposingRank = prefList.index(proposingMan)
        partnerRank = prefList.index(womanPartner)
        # partnerRank = 0
        # try:
        #     partnerRank = prefList.index(womanPartner)
        # except:
        #     return f"{woman} is currently unmatched, and should accept the proposal of man {proposingMan}"

        # print("PREFERENCE FUNCTION CALLED")
        # print((womanPref, proposingMan, woman, womanPartner))

        if proposingRank < partnerRank:
            return f"{woman} should break her engagement with man {womanPartner}, and accept the proposal of man {proposingMan}"
        else:
            return f"{woman} should keep her engagement with man {womanPartner}, and reject the proposal of man {proposingMan}"


    # # This step is automatically performed inside the AssistantAgent if the tool is a Python function.
    # preference_analysis_tool = FunctionTool(preference_analysis, description= """
    #                                         Use this function to determine if a matched woman (ex. W1) should accept a proposal from a man (ex. M1)
    #                                         If a woman has an existing partner and another man proposes to her, call this function. Use the following argments:
    #                                         womanPrefs should be of the form <[M1,M2,M3,M4,M5]>
    #                                         proposingMan should be a man of the form <M1>
    #                                         woman should be a woman of the form <W1>. This is the woman who is being proposed to
    #                                         womanPartner should be a man of the form <M1>. This is the woman's current partner
    #                                         """)
    # # The schema is provided to the model during AssistantAgent's on_messages call.
    # # preference_analysis_tool.schema

    # Create the primary agent.
    def get_proposer(self, instance):
        proposer = AssistantAgent(
            "Proposer",
            model_client=self.client,
            system_message="""
            You are a Proposer Agent in a multi-agent system implementing the Gale-Shapley algorithm for stable matching. Your role is to propose to Receiver Agents based on your preference list until you are matched or have no more options.

            ### **Your Attributes:**
            - **Unique Identifier:** You have a unique ID to distinguish yourself from other agents.
            - **Preference List:** A ranked list of Receiver Agents, from most preferred to least preferred, provided at initialization.
            - **Proposal Tracking:** You maintain a list of Receiver Agents you have already proposed to.
            - **Matching Status:** Initially unmatched; updated to matched when accepted by a Receiver Agent.
            - **Current Match:** The Receiver Agent you are currently matched to (if any).

            ### **Your Responsibilities:**
            - **On Receiving 'start_round' Message from Coordinator Agent:**
            - If you are unmatched:
                - Identify the next Receiver Agent on your preference list who you have not yet proposed to.
                - Send a 'proposal' message to that Receiver Agent, including your unique identifier.
            - If you are matched or have no remaining Receiver Agents to propose to, do nothing.
            - **On Receiving a Message from a Receiver Agent:**
            - If it is an 'accept' message:
                - Update your matching status to matched.
                - Record the Receiver Agent as your current match.
            - If it is a 'reject' message:
                - Do nothing; you will propose to the next Receiver Agent in the next round if unmatched.
            - **On Receiving 'status_check' Message from Coordinator Agent:**
            - Respond with:
                - Your current matching status (matched or unmatched).
                - Whether you have any remaining Receiver Agents to propose to (i.e., Receiver Agents on your preference list you haven’t proposed to yet).

            ### **Your Goal:**
            - Become matched to a Receiver Agent, ideally one as high as possible on your preference list.

            ### **Communication Notes:**
            - Include your unique identifier in all 'proposal' messages.
            - Expect responses ('accept' or 'reject') from Receiver Agents to include their identifier.
            """
        )

        return proposer

    def get_reciever(self, instance):
        reciever = AssistantAgent(
            "Reciever",
            model_client=self.client,
            system_message="""
            You are a Receiver Agent in a multi-agent system implementing the Gale-Shapley algorithm for stable matching. Your role is to evaluate proposals from Proposer Agents and accept the most preferred one based on your preference list.

            ### **Your Attributes:**
            - **Unique Identifier:** You have a unique ID to distinguish yourself from other agents.
            - **Preference List:** A ranked list of Proposer Agents, from most preferred to least preferred, provided at initialization.
            - **Matching Status:** Initially unmatched; updated to matched when you accept a Proposer Agent.
            - **Current Match:** The Proposer Agent you are currently matched to (if any).
            - **Current Round Proposals:** A list of proposals received in the current round.

            ### **Your Responsibilities:**
            - **On Receiving a 'proposal' Message from a Proposer Agent:**
            - Add the Proposer Agent’s identifier to your list of proposals for the current round.
            - **On Receiving an 'end_round' Message from Coordinator Agent:**
            - Evaluate:
                - All proposals received in this round.
                - Your current match (if you have one).
            - Select the Proposer Agent you prefer most based on your preference list from the combined set (current match + new proposals).
            - If you select a new Proposer Agent:
                - Send an 'accept' message to that Proposer Agent, including your identifier.
                - If you were previously matched, send a 'reject' message to your previous match.
                - Update your current match to the new Proposer Agent.
            - If you select your current match (i.e., no new proposal is preferred):
                - Send 'reject' messages to all Proposer Agents who proposed in this round.
            - Send 'reject' messages to all other Proposer Agents in the current round who were not selected.
            - Clear your list of proposals for the next round.

            ### **Your Goal:**
            - Be matched to the most preferred Proposer Agent possible, according to your preference list.

            ### **Communication Notes:**
            - Include your unique identifier in all 'accept' and 'reject' messages.
            - Expect 'proposal' messages to include the Proposer Agent’s identifier.
            """
        )

        return reciever

    def get_coordinator(self, instance):
        n = int(instance[1])

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"

        prefs = instance[3]

        coordinator = AssistantAgent(
            "Coordinator",
            model_client=self.client,
            system_message="""
            You are the Coordinator Agent in a multi-agent system implementing the Gale-Shapley algorithm for stable matching. Your role is to manage the rounds of the algorithm, synchronize the actions of Proposer and Receiver Agents, and determine when the process should terminate.

            ### **Your Responsibilities:**
            - **Initialization:**
            - Provide each Proposer Agent and Receiver Agent with their respective preference lists at the start of the process.
            - **Round Management:**
            - Start each round by sending a 'start_round' message to all Proposer Agents.
            - After all proposals are sent (assume a reasonable delay or confirmation mechanism in Autogen), send an 'end_round' message to all Receiver Agents.
            - After Receiver Agents have made their decisions, send a 'status_check' message to all Proposer Agents.
            - **Termination Check:**
            - Collect responses from all Proposer Agents to the 'status_check' message.
            - Check if any Proposer Agent is unmatched and still has Receiver Agents to propose to.
            - If at least one such Proposer Agent exists:
                - Initiate a new round by sending another 'start_round' message.
            - If no unmatched Proposer Agents have remaining options:
                - Terminate the process and announce that a stable matching has been achieved (e.g., by compiling and reporting the final matches from agent states).
                - Terminate the process by saying "TERMINATE"
            
            Once you have your final answer, Once you have found a stable matching, please return your matching in the JSON format given below:
            <answer>
            {FORMAT}
            </answer>

            Make sure that each man/woman is matched with exactly ONE partner. It is mandatory that you provide a matching as a JSON object enclosed in <answer></answer> tags as described above.

            ### **Your Goal:**
            - Ensure the algorithm executes correctly and terminates with a stable matching.

            ### **Communication Notes:**
            - Use broadcast messages ('start_round', 'end_round', 'status_check') to communicate with all relevant agents.
            - Expect responses to 'status_check' messages to include each Proposer Agent’s matching status and remaining options.

            Preferences:
            {PREF}
            """.format(FORMAT=json_format, PREF=prefs),
        )

        return coordinator

    def get_rrgc_list(self):
        # self.set_type_indices()
        # print(self.instances.values)
        # print("------------------")
        for i, instance in enumerate(self.instances.values):
            termination = TextMentionTermination("TERMINATE")
            coordinator = self.get_coordinator(instance)
            proposer = self.get_proposer(instance)
            reciever = self.get_reciever(instance)

            prompt = f"{coordinator._system_messages[0].content}\n=================================================\n{proposer._system_messages[0].content}\n=================================================\n{reciever._system_messages[0].content}\n\n"

            team = RoundRobinGroupChat(
                [coordinator, proposer, reciever],
                termination_condition=termination
            )
            # print(instance[self.type_index[type_]+1])
            self.rrgc_list.append((i, prompt, team, instance[3], instance[4], instance[5], instance[6], instance[1]))