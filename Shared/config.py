from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


##################################################################################################
###################################   JD Creation Models   #######################################
##################################################################################################
jd_creation_emb_llm = OpenAIEmbeddings(model=os.environ['JD_CREATION_EMB_MODEL'])
jd_creation_outline_llm = ChatOpenAI(model=os.environ['JD_CREATION_OUTLINE_LLM'], temperature=0.3)
jd_creation_llm_1 = ChatOpenAI(model=os.environ['JD_CREATION_LLM_1'], temperature=0.7)
jd_creation_llm_2 = ChatOpenAI(model=os.environ['JD_CREATION_LLM_2'], temperature=1.0)
jd_creation_llm_3 = ChatOpenAI(model=os.environ['JD_CREATION_LLM_3'], temperature=0.2)
jd_creation_judge_llm = ChatOpenAI(model=os.environ['JD_CREATION_JUDGE_LLM'], temperature=0.0)


##################################################################################################
###################################   AD Generation Models   #####################################
##################################################################################################
ad_generation_extract_llm= ChatOpenAI(model=os.environ['AD_GENERATION_LLM_EXTRACT'], temperature=0.2, max_tokens=2048)
ad_generation_brief_llm= ChatOpenAI(model=os.environ['AD_GENERATION_LLM_BRIEF'], temperature=0.3, max_tokens=16384)
ad_generation_llm= ChatOpenAI(model=os.environ['AD_GENERATION_LLM'], temperature=0.6, max_tokens=8192)

# Phase-based ad generation LLMs
# Variant A: skills-focused — lower temperature for precise, factual copy
ad_text_llm_a = ChatOpenAI(model=os.environ['AD_TEXT_LLM_A'], temperature=0.4, max_tokens=1024)
# Variant B: benefits-focused — higher temperature for warmer, emotive copy
ad_text_llm_b = ChatOpenAI(model=os.environ['AD_TEXT_LLM_B'], temperature=0.8, max_tokens=1024)
# Image script generator — moderate temperature, needs creativity but must stay on-brief
ad_image_script_llm = ChatOpenAI(model=os.environ['AD_IMAGE_SCRIPT_LLM'], temperature=0.6, max_tokens=512)