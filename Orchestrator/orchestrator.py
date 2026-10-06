import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from typing import List, Optional, Dict, Any
from datetime import datetime
from Shared.state import JDCreationState
from Create_JD.jd_creation_v2 import JD_Creation
from Generate_Ad.ad_generation_v2 import AD_Generation
from Post_Ad.ad_posting_v2 import AD_Posting
from langgraph.graph import StateGraph,END,START

class Orchestrator:
    def __init__(self):

        # Use in-memory checkpointer for better performance (no disk I/O, no freezing)
        self.checkpointer = MemorySaver(
            serde=JsonPlusSerializer()
        )
        self.conn = None  # No database connection needed for MemorySaver

        self.jd_agent = JD_Creation(checkpointer=self.checkpointer)
        self.ad_agent = AD_Generation(checkpointer=self.checkpointer)
        self.post_agent = AD_Posting(checkpointer=self.checkpointer)

        graph=StateGraph(JDCreationState)
        graph.add_node('jd_agent',self.jd_agent.run)
        graph.add_node('ad_agent',self.ad_agent.run)
        graph.add_node('post_agent',self.post_agent.run)

        graph.add_edge(START,'jd_agent')
        graph.add_conditional_edges('jd_agent',self.agent_decision,{
            'Jd Creation Complete':'ad_agent',
            'Ad Generation Complete':'post_agent',
            'Await Ad HR':END,
            'Await Ad Text HR':END,
            'Await Ad Image Script HR':END,
            'Await Ad Image HR':END,
            'Ad Posting Complete':END,
            'Error':END
        })
        graph.add_conditional_edges('ad_agent',self.agent_decision,{
            'Jd Creation Complete':'ad_agent',
            'Ad Generation Complete':'post_agent',
            'Await Ad HR':END,
            'Await Ad Text HR':END,
            'Await Ad Image Script HR':END,
            'Await Ad Image HR':END,
            'Ad Posting Complete':END,
            'Error':END
        })
        graph.add_conditional_edges('post_agent',self.agent_decision,{
            'Jd Creation Complete':'ad_agent',
            'Ad Generation Complete':'post_agent',
            'Await Ad HR':END,
            'Await Ad Text HR':END,
            'Await Ad Image Script HR':END,
            'Await Ad Image HR':END,
            'Ad Posting Complete':END,
            'Error':END
        })
        self.workflow=graph.compile(checkpointer=self.checkpointer)
    
    def run(self,initial_state:dict,thread_id:str):
        # Use thread_id for checkpointer
        result=self.workflow.invoke(initial_state,
        config={"configurable": {"thread_id": thread_id}},
        )
        return result

    def cleanup_db(self):
        """Clear in-memory checkpointer"""
        try:
            # MemorySaver stores data in memory, so we just need to clear it
            if hasattr(self.checkpointer, 'storage'):
                self.checkpointer.storage.clear()
                print(f"[INFO] Cleared in-memory checkpointer")
        except Exception as e:
            print(f"[WARNING] Failed to cleanup checkpointer: {e}")
    
    def agent_decision(self,state:JDCreationState):
        status = state.get("orchestration_status")
        if status in ('JD_Gen_Agent','await_hr','needs_review'):
            return 'Jd Creation Complete'
        elif status=='AD_Generation_Agent':
            return 'Ad Generation Complete'
        elif status == 'AD_AWAIT_HR':
            return 'Await Ad HR'
        elif status == 'AD_TEXT_AWAIT_HR':
            return 'Await Ad Text HR'
        elif status == 'AD_IMAGE_SCRIPT_AWAIT_HR':
            return 'Await Ad Image Script HR'
        elif status == 'AD_IMAGE_AWAIT_HR':
            return 'Await Ad Image HR'
        elif status in (
            'AD_Posting_Agent',
            'COLLECTING_APPLICATIONS',
            'SCREENING',
            'INTERVIEWS_DISPATCHED',
            'INTERVIEWS_COMPLETE',
            'OFFER_SENT',
            'PIPELINE_COMPLETE',
        ):
            return 'Ad Posting Complete'
        else:
            return "Error"