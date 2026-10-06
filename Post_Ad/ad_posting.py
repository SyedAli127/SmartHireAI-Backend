from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import sqlite3, uuid, os, requests
from dotenv import load_dotenv
import tweepy

load_dotenv()

# ==================== CONFIGURATION ==================== #

# Meta (Facebook & Instagram)
META_APP_ID = os.getenv("META_APP_ID")
META_APP_SECRET = os.getenv("META_APP_SECRET")
META_PAGE_ID = os.getenv("META_PAGE_ID")
META_INSTA_ID = os.getenv("META_INSTA_ID")
META_PAGE_TOKEN = os.getenv("META_PAGE_TOKEN")

# X (Twitter)
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET")

# ==================== X CLIENT SETUP ==================== #

def get_x_clients():

    try:
        if not all([X_BEARER_TOKEN, X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET]):
            return None, None
            
        client = tweepy.Client(
            bearer_token=X_BEARER_TOKEN,
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_SECRET
        )
        auth = tweepy.OAuth1UserHandler(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET)
        api_v1 = tweepy.API(auth)
        return client, api_v1
    except Exception as e:
        print(f"X (Twitter) Config Error: {e}")
        return None, None

x_client, x_api_v1 = get_x_clients()

# ==================== PYDANTIC SCHEMAS ==================== #

class PostResultSchema(BaseModel):
    platform: str
    status: Literal["success", "error", "skipped"]
    post_id: Optional[str] = None
    url: Optional[str] = None
    details: Optional[str] = None

class AdPostSummary(BaseModel):
    total_platforms: int
    successful: int
    failed: int
    results: List[PostResultSchema]

# ==================== STATE ==================== #

class AdPostingState(TypedDict):
    # Input
    message: str
    image_url: Optional[str]
    platforms: List[Literal["facebook", "instagram", "x"]]
    
    # Processing
    image_downloaded: Optional[bool]
    temp_image_path: Optional[str]
    
    # Results
    facebook_result: Optional[PostResultSchema]
    instagram_result: Optional[PostResultSchema]
    x_result: Optional[PostResultSchema]
    
    # Summary
    summary: Optional[AdPostSummary]
    post_id: Optional[str]
    created_at: Optional[str]
    status: Optional[Literal["running", "completed", "partial_failure", "failed"]]

# ==================== LLMs (if needed for optimization) ==================== #


# ==================== HELPER FUNCTIONS ==================== #

def download_image_for_x(image_url: str) -> Optional[str]:
    """Download image from URL for X posting"""
    if not image_url:
        return None
    
    try:
        print(f"   [INFO] Downloading image for X from: {image_url[:50]}...")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        temp_path = f"temp_x_{uuid.uuid4()}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        print(f"   [SUCCESS] Image downloaded: {temp_path}")
        return temp_path
    except Exception as e:
        print(f"   [ERROR] Failed to download image: {e}")
        return None

def cleanup_temp_image(image_path: Optional[str]):
    """Remove temporary image file"""
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"   [INFO] Cleaned up: {image_path}")
        except Exception as e:
            print(f"   [WARNING] Cleanup failed: {e}")

# ==================== NODE FUNCTIONS ==================== #

def validate_image_url(image_url: str) -> tuple[bool, str]:
    """Check if URL points to a valid image"""
    try:
        response = requests.head(image_url, allow_redirects=True, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        
        valid_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        
        if not any(t in content_type for t in valid_types):
            return False, f"URL returned Content-Type: '{content_type}', expected an image type"
        
        return True, content_type
    except Exception as e:
        return False, str(e)


def prepare_image_node(state: AdPostingState):
    """Node 1: Download image if needed for X"""
    image_url = state.get("image_url")
    platforms = state.get("platforms", [])
    
    if not image_url or "x" not in platforms:
        return {"image_downloaded": False, "temp_image_path": None}
    
    temp_path = download_image_for_x(image_url)
    
    return {
        "image_downloaded": bool(temp_path),
        "temp_image_path": temp_path
    }

def post_facebook_node(state: AdPostingState):
    """Node 2: Post to Facebook"""
    if "facebook" not in state.get("platforms", []):
        return {"facebook_result": PostResultSchema(
            platform="facebook",
            status="skipped",
            details="Not in selected platforms"
        )}
    
    message = state["message"]
    image_url = state.get("image_url")
    
    print("\n[FACEBOOK] Posting to Facebook...")
    
    if not META_PAGE_ID or not META_PAGE_TOKEN:
        return {"facebook_result": PostResultSchema(
            platform="facebook",
            status="error",
            details="Facebook credentials not configured"
        )}
    
    try:
        # Use photos endpoint if image, otherwise feed endpoint
        url = f"https://graph.facebook.com/v18.0/{META_PAGE_ID}/photos" if image_url else f"https://graph.facebook.com/v18.0/{META_PAGE_ID}/feed"
        
        payload = {
            "message": message,
            "access_token": META_PAGE_TOKEN
        }
        
        if image_url:
            payload["url"] = image_url
        
        response = requests.post(url, json=payload)
        data = response.json()
        
        if "id" in data:
            result = PostResultSchema(
                platform="facebook",
                status="success",
                post_id=data['id'],
                url=f"https://facebook.com/{data['id']}"
            )
            print(f"   [SUCCESS] Post ID: {data['id']}")
        else:
            result = PostResultSchema(
                platform="facebook",
                status="error",
                details=str(data)
            )
            print(f"   [ERROR] Failed: {data}")
        
        return {"facebook_result": result}
        
    except Exception as e:
        print(f"   [ERROR] Exception: {e}")
        return {"facebook_result": PostResultSchema(
            platform="facebook",
            status="error",
            details=str(e)
        )}

# def post_instagram_node(state: AdPostingState):
#     """Node 3: Post to Instagram"""
#     if "instagram" not in state.get("platforms", []):
#         return {"instagram_result": PostResultSchema(
#             platform="instagram",
#             status="skipped",
#             details="Not in selected platforms"
#         )}
    
#     message = state["message"]
#     image_url = state.get("image_url")
    
#     print("\n[INSTAGRAM] Posting to Instagram...")
    
#     if not META_INSTA_ID or not META_PAGE_TOKEN:
#         return {"instagram_result": PostResultSchema(
#             platform="instagram",
#             status="error",
#             details="Instagram credentials not configured"
#         )}
    
#     if not image_url:
#         return {"instagram_result": PostResultSchema(
#             platform="instagram",
#             status="error",
#             details="Instagram requires an image URL"
#         )}
    
#     try:
#         # Step 1: Create Container
#         print("   [INFO] Step 1/2: Creating container...")
#         container_url = f"https://graph.facebook.com/v18.0/{META_INSTA_ID}/media"
#         payload = {
#             "image_url": image_url,
#             "caption": message,
#             "access_token": META_PAGE_TOKEN
#         }
        
#         req = requests.post(container_url, json=payload)
#         data = req.json()
        
#         if "id" not in data:
#             result = PostResultSchema(
#                 platform="instagram",
#                 status="error",
#                 details=f"Container creation failed: {data}"
#             )
#             print(f"   [ERROR] Failed: {data}")
#             return {"instagram_result": result}
        
#         creation_id = data['id']
#         print(f"   [SUCCESS] Container created: {creation_id}")
        
#         # Step 2: Publish
#         print("   [INFO] Step 2/2: Publishing...")
#         publish_url = f"https://graph.facebook.com/v18.0/{META_INSTA_ID}/media_publish"
#         pub_payload = {
#             "creation_id": creation_id,
#             "access_token": META_PAGE_TOKEN
#         }
        
#         pub_req = requests.post(publish_url, json=pub_payload)
#         pub_data = pub_req.json()
        
#         if "id" in pub_data:
#             result = PostResultSchema(
#                 platform="instagram",
#                 status="success",
#                 post_id=pub_data['id'],
#                 url=f"https://instagram.com/p/{pub_data['id']}"
#             )
#             print(f"   [SUCCESS] Post ID: {pub_data['id']}")
#         else:
#             result = PostResultSchema(
#                 platform="instagram",
#                 status="error",
#                 details=f"Publish failed: {pub_data}"
#             )
#             print(f"   [ERROR] Publish failed: {pub_data}")
        
#         return {"instagram_result": result}
        
#     except Exception as e:
#         print(f"   [ERROR] Exception: {e}")
#         return {"instagram_result": PostResultSchema(
#             platform="instagram",
#             status="error",
#             details=str(e)
#         )}

def post_instagram_node(state: AdPostingState):
    """Node 3: Post to Instagram"""
    if "instagram" not in state.get("platforms", []):
        return {"instagram_result": PostResultSchema(
            platform="instagram",
            status="skipped",
            details="Not in selected platforms"
        )}
    
    message = state["message"]
    image_url = state.get("image_url")
    
    print("\n[INSTAGRAM] Posting to Instagram...")
    
    if not META_INSTA_ID or not META_PAGE_TOKEN:
        return {"instagram_result": PostResultSchema(
            platform="instagram",
            status="error",
            details="Instagram credentials not configured"
        )}
    
    if not image_url:
        return {"instagram_result": PostResultSchema(
            platform="instagram",
            status="error",
            details="Instagram requires an image URL"
        )}
    
    try:
        # Step 1: Create Container
        print("   [INFO] Step 1/2: Creating container...")
        req = requests.post(
            f"https://graph.facebook.com/v18.0/{META_INSTA_ID}/media",
            json={
                "image_url": image_url,
                "caption": message,
                "access_token": META_PAGE_TOKEN
            }
        )
        data = req.json()
        
        if "id" not in data:
            print(f"   [ERROR] Failed: {data}")
            return {"instagram_result": PostResultSchema(
                platform="instagram",
                status="error",
                details=f"Container creation failed: {data}"
            )}
        
        creation_id = data['id']
        print(f"   [SUCCESS] Container created: {creation_id}")
        
        # Step 2: Publish
        print("   [INFO] Step 2/2: Publishing...")
        pub_req = requests.post(
            f"https://graph.facebook.com/v18.0/{META_INSTA_ID}/media_publish",
            json={
                "creation_id": creation_id,
                "access_token": META_PAGE_TOKEN
            }
        )
        pub_data = pub_req.json()
        
        if "id" not in pub_data:
            print(f"   [ERROR] Publish failed: {pub_data}")
            return {"instagram_result": PostResultSchema(
                platform="instagram",
                status="error",
                details=f"Publish failed: {pub_data}"
            )}

        post_id = pub_data['id']

        # Step 3: Fetch real permalink
        print("   [INFO] Fetching permalink...")
        permalink_resp = requests.get(
            f"https://graph.facebook.com/v18.0/{post_id}",
            params={
                "fields": "permalink",
                "access_token": META_PAGE_TOKEN
            }
        ).json()
        
        real_url = permalink_resp.get("permalink", "https://www.instagram.com/")
        
        print(f"   [SUCCESS] Post ID: {post_id}")
        print(f"   [SUCCESS] URL: {real_url}")
        
        return {"instagram_result": PostResultSchema(
            platform="instagram",
            status="success",
            post_id=post_id,
            url=real_url  # ✅ real clickable URL
        )}
        
    except Exception as e:
        print(f"   [ERROR] Exception: {e}")
        return {"instagram_result": PostResultSchema(
            platform="instagram",
            status="error",
            details=str(e)
        )}

def post_x_node(state: AdPostingState):
    """Node 4: Post to X (Twitter)"""
    if "x" not in state.get("platforms", []):
        return {"x_result": PostResultSchema(
            platform="x",
            status="skipped",
            details="Not in selected platforms"
        )}
    
    message = state["message"]
    temp_image_path = state.get("temp_image_path")
    
    print("\n[X] Posting to X (Twitter)...")
    
    if not x_client or not x_api_v1:
        return {"x_result": PostResultSchema(
            platform="x",
            status="error",
            details="X credentials not configured"
        )}
    
    try:
        if temp_image_path and os.path.exists(temp_image_path):
            print("   [INFO] Uploading image...")
            # Upload media using v1.1 API
            media = x_api_v1.media_upload(temp_image_path)
            # Post with media
            status = x_api_v1.update_status(status=message, media_ids=[media.media_id])
            result = PostResultSchema(
                platform="x",
                status="success",
                post_id=str(status.id),
                url=f"https://x.com/user/status/{status.id}"
            )
            print(f"   [SUCCESS] Tweet ID: {status.id}")
        else:
            print("   [INFO] Posting text-only...")
            # Text-only tweet using v2 API
            response = x_client.create_tweet(text=message)
            result = PostResultSchema(
                platform="x",
                status="success",
                post_id=response.data['id'],
                url=f"https://x.com/user/status/{response.data['id']}"
            )
            print(f"   [SUCCESS] Tweet ID: {response.data['id']}")
        
        return {"x_result": result}
        
    except Exception as e:
        print(f"   [ERROR] Exception: {e}")
        return {"x_result": PostResultSchema(
            platform="x",
            status="error",
            details=str(e)
        )}

def cleanup_node(state: AdPostingState):
    """Node 5: Cleanup temporary files"""
    temp_path = state.get("temp_image_path")
    if temp_path:
        cleanup_temp_image(temp_path)
    return {}

def summary_node(state: AdPostingState):
    """Node 6: Generate summary and final status"""
    results = []
    
    # Collect all results
    if state.get("facebook_result"):
        results.append(state["facebook_result"])
    if state.get("instagram_result"):
        results.append(state["instagram_result"])
    if state.get("x_result"):
        results.append(state["x_result"])
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "error")
    
    # Determine overall status
    if successful == total:
        overall_status = "completed"
    elif successful > 0:
        overall_status = "partial_failure"
    else:
        overall_status = "failed"
    
    summary = AdPostSummary(
        total_platforms=total,
        successful=successful,
        failed=failed,
        results=results
    )
    
    return {
        "summary": summary,
        "post_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": overall_status
    }

# ==================== GRAPH SETUP ==================== #

conn = sqlite3.connect("ad_posting.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(AdPostingState)

# Add nodes
graph.add_node("prepare_image", prepare_image_node)
graph.add_node("post_facebook", post_facebook_node)
graph.add_node("post_instagram", post_instagram_node)
graph.add_node("post_x", post_x_node)
graph.add_node("cleanup", cleanup_node)
graph.add_node("summary", summary_node)

# Add edges
graph.add_edge(START, "prepare_image")
graph.add_edge("prepare_image", "post_facebook")
graph.add_edge("prepare_image", "post_instagram")
graph.add_edge("prepare_image", "post_x")

graph.add_edge("post_facebook", "cleanup")
graph.add_edge("post_instagram", "cleanup")
graph.add_edge("post_x", "cleanup")

graph.add_edge("cleanup", "summary")
graph.add_edge("summary", END)

ad_posting_workflow = graph.compile(checkpointer=checkpointer)

# ==================== RUN ==================== #

def post_ad(message: str, image_url: Optional[str] = None, platforms: List[str] = None):
    """Main function to post ad to multiple platforms"""
    
    if platforms is None:
        platforms = ["facebook", "instagram", "x"]
    
    # Normalize platform names
    normalized_platforms = []
    for p in platforms:
        p_lower = p.lower().strip()
        if p_lower in ["fb", "facebook"]:
            normalized_platforms.append("facebook")
        elif p_lower in ["ig", "insta", "instagram"]:
            normalized_platforms.append("instagram")
        elif p_lower in ["x", "twitter"]:
            normalized_platforms.append("x")
    if image_url:
        is_valid, reason = validate_image_url(image_url)
        if not is_valid:
            print(f"\n[WARNING] Image URL rejected: {reason}")
            print("[INFO] Proceeding without image (text-only)...")
            image_url = None  # fallback to text-only
    initial_state = {
        "message": message,
        "image_url": image_url,
        "platforms": normalized_platforms,
        "status": "running"
    }
    
    thread_id = str(uuid.uuid4())
    

    print(f"Message: {message[:50]}...")
    print(f"Image: {image_url[:50] if image_url else 'None'}...")
    print(f"Platforms: {', '.join(normalized_platforms)}")
    print("="*60)
    
    result = ad_posting_workflow.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    return result

def display_results(result: Dict[str, Any]):
    """Display posting results in a formatted way"""
    print("\n" + "="*60)
    print("POSTING RESULTS")
    print("="*60)
    
    summary = result.get("summary")
    if summary:
        print(f"\nSuccessful: {summary.successful}/{summary.total_platforms}")
        print(f"Failed: {summary.failed}/{summary.total_platforms}")
        print(f"Created: {result.get('created_at')}")
        print(f"Post ID: {result.get('post_id')}")
        print(f"Status: {result.get('status').upper()}")
        
        print("\n" + "-"*60)
        print("PLATFORM DETAILS:")
        print("-"*60)
        
        for res in summary.results:
            status_label = "[SUCCESS]" if res.status == "success" else "[ERROR]" if res.status == "error" else "[SKIPPED]"
            print(f"\n{status_label} {res.platform.upper()}")
            print(f"   Status: {res.status}")
            if res.url:
                print(f"   URL: {res.url}")
            if res.post_id:
                print(f"   Post ID: {res.post_id}")
            if res.details:
                print(f"   Details: {res.details}")
    
    print("\n" + "="*60)

# ==================== MAIN FUNCTION ==================== #

def main():
    """Main function to run the ad posting agent"""
    import sys
    
    # Check if we have ads to post
    ad_file_path = "Generated_Ads"
    
    if os.path.exists(ad_file_path) and os.path.isdir(ad_file_path):
        # Look for generated ads
        ad_files = [f for f in os.listdir(ad_file_path) if f.endswith('.txt') and not f.endswith('_image.txt') and not f.endswith('_video.txt')]
        
        if ad_files:
            print("\n[INFO] Found generated ads!")
            print(f"Total ads: {len(ad_files)}")
            
            # Ask which ad to post
            print("\nAvailable ads:")
            for idx, ad_file in enumerate(ad_files, 1):
                print(f"   {idx}. {ad_file}")
            
            choice = input("\nSelect ad number to post (or 'custom' for manual input): ").strip()
            
            if choice.lower() == 'custom':
                # Manual input mode
                pass  # Will fall through to manual input below
            elif choice.isdigit() and 1 <= int(choice) <= len(ad_files):
                # Read selected ad file
                selected_ad = ad_files[int(choice) - 1]
                ad_path = os.path.join(ad_file_path, selected_ad)
                
                with open(ad_path, 'r', encoding='utf-8') as f:
                    ad_content = f.read()
                
                print(f"\n[INFO] Selected Ad: {selected_ad}")
                print("-"*60)
                print(ad_content)
                print("-"*60)
                
                # Extract message (simplified parsing)
                lines = ad_content.split('\n')
                message_lines = []
                capture = False
                for line in lines:
                    if line.startswith('Headline:'):
                        continue
                    elif line.startswith('CTA:'):
                        break
                    elif line.startswith('Platform:') or line.startswith('Variant:'):
                        continue
                    elif line.strip() and not line.startswith('Hashtags:'):
                        if not capture and not line.startswith('Platform:') and not line.startswith('Variant:') and not line.startswith('Headline:'):
                            capture = True
                        if capture:
                            message_lines.append(line)
                
                message = '\n'.join(message_lines).strip()
                
                # Ask for image URL
                image_url = input("\nEnter image URL (optional, press Enter to skip): ").strip() or None
                
                # Ask for platforms
                print("\nSelect platforms (comma-separated):")
                print("   Options: facebook, instagram, x")
                platforms_input = input("   Selection [all]: ").strip().lower() or "all"
                
                if platforms_input == "all":
                    platforms = ["facebook", "instagram", "x"]
                else:
                    platforms = [p.strip() for p in platforms_input.split(",")]
                
                # Post the ad
                result = post_ad(message, image_url, platforms)
                display_results(result)
                
                return
    
    # Manual input mode
    print("\n" + "="*60)
    print("MANUAL AD INPUT MODE")
    print("="*60)
    
    message = input("\nEnter your ad message:\n").strip()
    
    if not message:
        print("[ERROR] Message cannot be empty!")
        sys.exit(1)
    
    image_url = input("\nEnter image URL (optional, press Enter to skip): ").strip() or None
    
    print("\nSelect platforms (comma-separated):")
    print("   Options: facebook, instagram, x")
    platforms_input = input("   Selection [all]: ").strip().lower() or "all"
    
    if platforms_input == "all":
        platforms = ["facebook", "instagram", "x"]
    else:
        platforms = [p.strip() for p in platforms_input.split(",")]
    
    # Post the ad
    result = post_ad(message, image_url, platforms)
    display_results(result)

if __name__ == "__main__":
    main()
