import os
import datetime,time
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from datetime import timezone  
from email.message import EmailMessage
import base64
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

class AgentState(TypedDict):
    candidate_name: str
    candidate_email: str
    interviewer_email: str
    suggested_slots: List[str]
    raw_email_reply: Optional[str]
    status: str

def route_start(state: AgentState):
    # If we have a reply, we are in Beat 2 (Booking)
    if state.get("raw_email_reply"):
        return "book"
    # Otherwise, we are in Beat 1 (Invitation)
    return "invite"

# --- API INITIALIZERS ---
def get_services():
    """Get Google API services, generating token.json if necessary."""
    token_path = 'tokens/token.json'
    scopes = [
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'  # Changed from readonly to modify        
    ]
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'secret/client_secret_2_984160960111-pg7ed2oc52dttic5oi11rtc13a5qtsqu.apps.googleusercontent.com.json',
            scopes=scopes
        )
        creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds), build('gmail', 'v1', credentials=creds)

# --- NODES ---

def send_invitation(state: AgentState):
    cal_service, gmail_service = get_services()
    
    # Validate that 'interviewer_email' exists in the state
    if 'interviewer_email' not in state:
        raise KeyError("The 'interviewer_email' key is missing from the state dictionary.")

    # 1. Define the Window (e.g., next 3 business days)
    now = datetime.datetime.now(timezone.utc)
    search_end = now + datetime.timedelta(days=3)
    
    fb_query = {
        "timeMin": now.isoformat(),
        "timeMax": search_end.isoformat(),
        "items": [{"id": state['interviewer_email']}]
    }
    fb_res = cal_service.freebusy().query(body=fb_query).execute()
    busy_slots = fb_res['calendars'][state['interviewer_email']]['busy']
    
    # 2. Dynamic Gap Finding
    # We want to find 1-hour gaps between 9 AM and 5 PM
    suggested = []
    current_check = now.replace(hour=9, minute=0, second=0, microsecond=0)
    
    while len(suggested) < 3 and current_check < search_end:
        # Move to next day if current_check is past 5 PM
        if current_check.hour >= 17:
            current_check += datetime.timedelta(days=1)
            current_check = current_check.replace(hour=9, minute=0)
            continue

        slot_start = current_check
        slot_end = current_check + datetime.timedelta(hours=1)
        
        # Check if this 1-hour block overlaps with any 'busy' time
        is_free = True
        for busy in busy_slots:
            b_start = datetime.datetime.fromisoformat(busy['start'].replace('Z', '+00:00'))
            b_end = datetime.datetime.fromisoformat(busy['end'].replace('Z', '+00:00'))
            # Overlap logic
            if not (slot_end <= b_start or slot_start >= b_end):
                is_free = False
                break
        
        if is_free:
            suggested.append(slot_start.isoformat().replace('+00:00', 'Z'))
            current_check += datetime.timedelta(hours=2) # Space out suggestions
        else:
            current_check += datetime.timedelta(minutes=30) # Increment and check again

    # 3. Update State with REAL slots
    state['suggested_slots'] = suggested  # Ensure this key is set
    
    
    # 2. Construct the Email
    readable_slots = "\n".join([
        datetime.datetime.fromisoformat(s.replace('Z', '')).strftime('%A, %b %d at %I:%M %p')
        for s in suggested
    ])

    email_content = f"Hi {state['candidate_name']},\n\nYou've been shortlisted! Please reply with your preferred choice from the following slots:\n\n{readable_slots}\n\nBest regards,\nRecruitment Team"

    mime_msg = EmailMessage()
    mime_msg.set_content(email_content)
    mime_msg['To'] = state['candidate_email']
    mime_msg['Subject'] = f"Interview Invitation: {state['candidate_name']}"
    mime_msg['From'] = state['interviewer_email']

    # 3. Encode and Send via REAL Gmail API
    try:
        encoded_message = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        
        # This is the line that actually sends the email
        gmail_service.users().messages().send(userId="me", body=create_message).execute()
        
        print(f"--- [REAL GMAIL] Invitation successfully sent to {state['candidate_email']} ---")
    except Exception as e:
        print(f"--- [ERROR] Failed to send email: {e} ---")
        return {"status": "FAILED"}
    
    return {"suggested_slots": suggested, "status": "AWAITING_REPLY"}

def run_inbox_listener(thread_id, candidate_email):
    """
    Polls Gmail for a reply and triggers the 'book' node.
    """
    _, gmail_service = get_services()
    print(f"--- [LISTENER] Watching for reply from {candidate_email}... ---")

    while True:
        # 1. Search for unread messages from the candidate
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Checking inbox for {candidate_email}...")
        query = f"from:{candidate_email} is:unread"
        results = gmail_service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])

        if messages:
            msg = gmail_service.users().messages().get(userId='me', id=messages[0]['id']).execute()
            email_body = msg['snippet'] # The candidate's reply text
            
            print(f"--- [REPLY DETECTED] Body: {email_body} ---")

            # 2. Resume the LangGraph with the new data
            app.invoke({
                "raw_email_reply": email_body,
                "status": "AWAITING_REPLY"
            }, config={"configurable": {"thread_id": thread_id}})
            
            # 3. Mark as read so we don't process it twice
            gmail_service.users().messages().batchModify(
                userId='me', 
                body={'addLabelIds': [], 'removeLabelIds': ['UNREAD'], 'ids': [messages[0]['id']]}
            ).execute()
            
            break # Exit loop once booked
        
        time.sleep(10) # Check every 30 seconds

def finalize_booking(state: AgentState):
    """Parses the reply, matches it to a slot, and creates the Google Meet."""
    cal_service, _ = get_services()
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    # Use local time for context instead of UTC
    local_now = datetime.datetime.now().astimezone() 
    current_context = local_now.strftime("%A, %B %d, %Y %I:%M %p %Z")

    # Update the prompt template to match the provided input variables
    prompt = ChatPromptTemplate.from_template(
        "Candidate said: '{reply}'. From these available ISO slots: {slots}, which did they choose? "
        "Output ONLY the ISO timestamp. If no match, output 'REJECT'."
    )
    chain = prompt | llm
    prediction = chain.invoke({"reply": state['raw_email_reply'], "slots": state['suggested_slots']})
    chosen_iso = prediction.content.strip()

    if "REJECT" in chosen_iso:
        print("--- [AGENT] No valid slot found in reply. ---")
        return {"status": "FAILED"}

    # Add validation to check for 'suggested_slots'
    if 'suggested_slots' not in state:
        raise KeyError("The 'suggested_slots' key is missing from the state dictionary.")

    # Create GCal Event with Google Meet
    event = {
        'summary': f'Interview: {state["candidate_name"]}',
        'start': {'dateTime': chosen_iso, 'timeZone': 'UTC'},
        'end': {'dateTime': (datetime.datetime.fromisoformat(chosen_iso.replace('Z','')) + datetime.timedelta(hours=1)).isoformat() + 'Z'},
        'attendees': [{'email': state['candidate_email']}, {'email': state['interviewer_email']}],
        'conferenceData': {'createRequest': {'requestId': f"meet-{chosen_iso}", 'conferenceSolutionKey': {'type': 'hangoutsMeet'}}}
    }
    
    created_event = cal_service.events().insert(
        calendarId='primary', body=event, conferenceDataVersion=1
    ).execute()

    print(f"--- [SUCCESS] Event created: {created_event.get('htmlLink')} ---")
    return {"status": "COMPLETED"}

# --- GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("invite", send_invitation)
builder.add_node("book", finalize_booking)

builder.set_conditional_entry_point(
    route_start,
    {
        "invite": "invite",
        "book": "book"
    }
)
builder.add_edge("invite", END) 
builder.add_edge("book", END)

# Add the memory saver here
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    # 1. Setup metadata
    candidate_email = "syedali3940@gmail.com"
    # Use a dynamic thread_id so you can run multiple tests
    unique_thread_id = f"interview_{int(time.time())}" 
    config = {"configurable": {"thread_id": unique_thread_id}}

    print(f"--- [STARTING AGENT] Thread: {unique_thread_id} ---")

    # 2. BEAT 1: Send the Invitation
    # This will run the 'invite' node and then return
    initial_state = {
        "candidate_name": "Ali",
        "candidate_email": candidate_email,
        "interviewer_email": "syed.ali@kodexolabs.com",  # Add this key explicitly
        "suggested_slots": [],
        "raw_email_reply": None,
        "status": ""
    }

    app.invoke(initial_state, config)

    # 3. BEAT 2: The Listener (The "Stay Alive" part)
    # This function has a 'while True' loop, which keeps the script running
    try:
        run_inbox_listener(unique_thread_id, candidate_email)
    except KeyboardInterrupt:
        print("\n--- [STOPPING AGENT] Listener turned off. ---")