from langgraph.graph import START, StateGraph, END
from typing import Dict, Any, List, Optional
import os, uuid, time
from Shared.state import JDCreationState
from Shared.schemas import PostResultSchema, AdPostSummary
import requests
import tweepy
from dotenv import load_dotenv

load_dotenv()

META_PAGE_ID = os.getenv("META_PAGE_ID")
META_INSTA_ID = os.getenv("META_INSTA_ID")
META_PAGE_TOKEN = os.getenv("META_PAGE_TOKEN")

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET")


def _ad_get(ad, key: str, default=""):
    """Get a field from either a Pydantic model or a plain dict."""
    if isinstance(ad, dict):
        return ad.get(key, default)
    return getattr(ad, key, default) or default


def _ad_find(approved_ads, platform: str):
    """Return the first ad whose platform matches, handling Pydantic or dict."""
    for ad in approved_ads:
        p = ad.get("platform") if isinstance(ad, dict) else getattr(ad, "platform", None)
        if p == platform:
            return ad
    return None


class AD_Posting:
    def __init__(self, checkpointer):
        """Initialize Ad Posting agent with graph workflow"""
        
        self.x_client, self.x_api_v1 = self._init_x_clients()
        
        graph = StateGraph(JDCreationState)
        graph.add_node("prepare_image", self.prepare_image_node)
        graph.add_node("post_facebook", self.post_facebook_node)
        graph.add_node("post_instagram", self.post_instagram_node)
        graph.add_node("post_x", self.post_x_node)
        graph.add_node("collect_results", self.collect_results_node)
        graph.add_node("cleanup", self.cleanup_node)
        graph.add_node("summary", self.summary_node)

        graph.add_edge(START, "prepare_image")
        graph.add_edge("prepare_image", "post_facebook")
        graph.add_edge("prepare_image", "post_instagram")
        graph.add_edge("prepare_image", "post_x")
        
        # All posting nodes go to collect_results
        graph.add_edge("post_facebook", "collect_results")
        graph.add_edge("post_instagram", "collect_results")
        graph.add_edge("post_x", "collect_results")
        
        graph.add_edge("collect_results", "cleanup")
        graph.add_edge("cleanup", "summary")
        graph.add_edge("summary", END)

        # Don't use nested checkpointer - let parent orchestrator handle it
        self.ad_posting_workflow = graph.compile()

    def _init_x_clients(self):
        """Initialize X (Twitter) clients"""
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
            print(f"[WARNING] X (Twitter) Config Error: {e}")
            return None, None

    def run(self, state: JDCreationState):
        """Run ad posting workflow"""
        # No need for thread_id config since we're not using nested checkpointing
        result = self.ad_posting_workflow.invoke(state)
        return result

    def _image_url(self, image_path: str) -> Optional[str]:
        """Return a public image URL. S3 URLs pass through; local paths use static mount."""
        if not image_path:
            return None
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return image_path
        api_base = os.getenv("API_BASE_URL", "").rstrip("/")
        if not api_base or "localhost" in api_base or "127.0.0.1" in api_base:
            return None
        folder = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)
        return f"{api_base}/generated_ads/{folder}/{filename}"

    def _download_image_for_x(self, image_url: str) -> str:
        """Download image from URL to a temp local file for X media upload."""
        if not image_url:
            return None
        try:
            import tempfile
            print(f"   [INFO] Downloading image for X from: {image_url[:50]}...")
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            suffix = ".jpg" if "jpg" in image_url.lower() else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)
                temp_path = f.name
            print(f"   [SUCCESS] Image downloaded to temp file")
            return temp_path
        except Exception as e:
            print(f"   [ERROR] Failed to download image: {e}")
            return None

    def _cleanup_temp_image(self, image_path: str):
        """Remove temporary image file."""
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"   [INFO] Cleaned up temp file")
            except Exception as e:
                print(f"   [WARNING] Cleanup failed: {e}")

    def prepare_image_node(self, state: JDCreationState):
        """Node 1: Prepare for posting (check for images, etc.)"""
        output_dir = state.get("output_dir")
        platforms = state.get("platforms", [])
        
        print("\n[PREPARE] Starting ad posting preparation...")
        print(f"[PREPARE] Target platforms: {platforms}")
        
        if "X" in platforms and output_dir and os.path.exists(output_dir):
            image_files = [f for f in os.listdir(output_dir) if f.endswith('_image.txt')]
            if image_files:
                print(f"   [INFO] Found image prompt files: {len(image_files)}")
        
        # Don't initialize posting_urls - let the posting nodes add to it
        return {}

    def post_facebook_node(self, state: JDCreationState):
        platforms = state.get("platforms", [])
        if "Facebook" not in platforms:
            return {}

        print("\n[FACEBOOK] Posting to Facebook...")

        if not META_PAGE_ID or not META_PAGE_TOKEN:
            return {"posting_urls": {"Facebook": "Error: Credentials not configured"}}

        approved_ads = state.get("approved_ads", [])
        fb_ad = _ad_find(approved_ads, "Facebook")

        if not fb_ad:
            return {"posting_urls": {"Facebook": "Error: No approved ad found"}}

        message = f"{_ad_get(fb_ad, 'headline')}\n\n{_ad_get(fb_ad, 'text')}\n\n"
        message += f"{_ad_get(fb_ad, 'cta')}\n\n"
        message += " ".join(_ad_get(fb_ad, 'hashtags') or [])
        message += f"\n\nApply here: {state.get('job_form_link', '')}"

        image_path = _ad_get(fb_ad, "generated_image_path")
        image_url = self._image_url(image_path)

        for attempt in range(3):
            if attempt > 0:
                print(f"   [INFO] Retrying Facebook post (attempt {attempt + 1}/3)...")
                time.sleep(10)
            try:
                if image_url:
                    print(f"   [INFO] Posting with image URL: {image_url[:60]}...")
                    url = f"https://graph.facebook.com/v18.0/{META_PAGE_ID}/photos"
                    response = requests.post(
                        url,
                        data={
                            "url": image_url,
                            "caption": message,
                            "access_token": META_PAGE_TOKEN,
                            "published": "true",
                        },
                    )
                elif image_path and os.path.exists(image_path):
                    print(f"   [INFO] Uploading local image: {image_path}")
                    url = f"https://graph.facebook.com/v18.0/{META_PAGE_ID}/photos"
                    with open(image_path, "rb") as img_file:
                        response = requests.post(
                            url,
                            data={
                                "caption": message,
                                "access_token": META_PAGE_TOKEN,
                                "published": "true",
                            },
                            files={"source": img_file},
                        )
                else:
                    print("   [INFO] No image found, posting text only")
                    url = f"https://graph.facebook.com/v18.0/{META_PAGE_ID}/feed"
                    response = requests.post(url, json={
                        "message": message,
                        "access_token": META_PAGE_TOKEN,
                    })

                data = response.json()
                if "id" in data:
                    post_url = f"https://facebook.com/{data['id']}"
                    print(f"   [SUCCESS] URL: {post_url}")
                    return {"posting_urls": {"Facebook": post_url}}

                error = data.get("error", {})
                print(f"   [ERROR] {error.get('message', 'Unknown error')}")
                # Don't retry non-transient errors (auth, permissions, etc.)
                if error.get("code") not in (1, 2, 4, 17, 341):
                    return {"posting_urls": {"Facebook": f"Error: {error.get('message', 'Unknown error')}"}}

            except Exception as e:
                print(f"   [ERROR] Exception: {e}")
                if attempt == 2:
                    return {"posting_urls": {"Facebook": f"Error: {str(e)}"}}

        return {"posting_urls": {"Facebook": "Error: Failed after 3 attempts"}}

    # def post_instagram_node(self, state: JDCreationState):
    #     """Node 3: Post to Instagram"""
    #     platforms = state.get("platforms", [])
        
    #     if "Instagram" not in platforms:
    #         return {}
        
    #     print("\n[INSTAGRAM] Posting to Instagram...")
        
    #     if not META_INSTA_ID or not META_PAGE_TOKEN:
    #         print("   [ERROR] Instagram credentials not configured")
    #         return {"posting_urls": {"Instagram": "Error: Credentials not configured"}}
        
    #     approved_ads = state.get("approved_ads", [])
    #     ig_ad = next((ad for ad in approved_ads if ad.get("platform") == "Instagram"), None)
        
    #     if not ig_ad:
    #         print("   [ERROR] No approved Instagram ad found")
    #         return {"posting_urls": {"Instagram": "Error: No approved ad found"}}
        
    #     print("   [WARNING] Instagram requires an image URL. Skipping for now.")
    #     print("   [INFO] To post to Instagram, provide image_url in state")
        
    #     return {"posting_urls": {"Instagram": "Skipped: Requires image URL"}}
    
    def _upload_image_to_facebook_for_instagram(self, image_path: str) -> str:
        """Upload image to Facebook unpublished, get a hosted URL for Instagram."""
        try:
            url = f"https://graph.facebook.com/v18.0/{META_PAGE_ID}/photos"
            with open(image_path, "rb") as img_file:
                response = requests.post(
                    url,
                    data={
                        "access_token": META_PAGE_TOKEN,
                        "published": "false",  # Don't publish to FB, just host it
                    },
                    files={"source": img_file},
                )
            data = response.json()
            if "id" not in data:
                print(f"   [ERROR] Could not upload image for Instagram: {data}")
                return None

            # Fetch the hosted picture URL
            photo_id = data["id"]
            fields_resp = requests.get(
                f"https://graph.facebook.com/v18.0/{photo_id}",
                params={"fields": "images", "access_token": META_PAGE_TOKEN},
            ).json()

            images = fields_resp.get("images", [])
            if images:
                return images[0]["source"]  # Largest image URL
            return None
        except Exception as e:
            print(f"   [ERROR] Image upload for Instagram failed: {e}")
            return None
        
    def post_instagram_node(self, state: JDCreationState):
        platforms = state.get("platforms", [])
        if "Instagram" not in platforms:
            return {}

        print("\n[INSTAGRAM] Posting to Instagram...")

        if not META_INSTA_ID or not META_PAGE_TOKEN:
            return {"posting_urls": {"Instagram": "Error: Credentials not configured"}}

        approved_ads = state.get("approved_ads", [])
        ig_ad = _ad_find(approved_ads, "Instagram")

        if not ig_ad:
            return {"posting_urls": {"Instagram": "Error: No approved ad found"}}

        image_path = _ad_get(ig_ad, "generated_image_path")
        image_url = self._image_url(image_path) or state.get("image_url")

        if not image_url and image_path and os.path.exists(image_path):
            print("   [INFO] Uploading local image to Facebook CDN for Instagram...")
            image_url = self._upload_image_to_facebook_for_instagram(image_path)

        if not image_url:
            print("   [ERROR] No usable image URL for Instagram")
            return {"posting_urls": {"Instagram": "Error: No image URL available"}}

        caption = f"{_ad_get(ig_ad, 'headline')}\n\n{_ad_get(ig_ad, 'text')}\n\n"
        caption += " ".join(_ad_get(ig_ad, 'hashtags') or [])
        caption += f"\n\nApply here: {state.get('job_form_link', '')}"

        try:
            # Step 1: Create container
            print("   [INFO] Step 1/2: Creating container...")
            req = requests.post(
                f"https://graph.facebook.com/v18.0/{META_INSTA_ID}/media",
                json={
                    "image_url": image_url,
                    "caption": caption,
                    "access_token": META_PAGE_TOKEN
                }
            )
            data = req.json()

            if "id" not in data:
                print(f"   [ERROR] Container failed: {data}")
                return {"posting_urls": {"Instagram": f"Error: {data.get('error', {}).get('message', 'Container failed')}"}}

            creation_id = data['id']
            print(f"   [SUCCESS] Container created: {creation_id}")

            # Wait for container to be ready (Instagram processes media asynchronously)
            print("   [INFO] Waiting for container to be ready...")
            for attempt in range(10):
                time.sleep(5)
                status_resp = requests.get(
                    f"https://graph.facebook.com/v18.0/{creation_id}",
                    params={"fields": "status_code", "access_token": META_PAGE_TOKEN}
                ).json()
                status_code = status_resp.get("status_code", "")
                print(f"   [INFO] Container status: {status_code} (attempt {attempt + 1}/10)")
                if status_code == "FINISHED":
                    break
                if status_code == "ERROR":
                    return {"posting_urls": {"Instagram": "Error: Container processing failed"}}
            else:
                return {"posting_urls": {"Instagram": "Error: Container timed out (still processing)"}}

            # Step 2: Publish (with retries)
            print("   [INFO] Step 2/2: Publishing...")
            pub_data = None
            for pub_attempt in range(3):
                if pub_attempt > 0:
                    print(f"   [INFO] Retrying publish (attempt {pub_attempt + 1}/3)...")
                    time.sleep(10)
                pub_req = requests.post(
                    f"https://graph.facebook.com/v18.0/{META_INSTA_ID}/media_publish",
                    json={
                        "creation_id": creation_id,
                        "access_token": META_PAGE_TOKEN
                    }
                )
                pub_data = pub_req.json()
                if "id" in pub_data:
                    break
                err_code = pub_data.get("error", {}).get("error_subcode")
                print(f"   [ERROR] Publish failed: {pub_data}")
                # Only retry transient errors (2207027 = media not ready)
                if err_code != 2207027:
                    break

            if "id" not in pub_data:
                return {"posting_urls": {"Instagram": f"Error: {pub_data.get('error', {}).get('message', 'Publish failed')}"}}

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

            return {"posting_urls": {"Instagram": real_url}}

        except Exception as e:
            print(f"   [ERROR] Exception: {e}")
            return {"posting_urls": {"Instagram": f"Error: {str(e)}"}}

    def post_x_node(self, state: JDCreationState):
        """Node 4: Post to X (Twitter)"""
        platforms = state.get("platforms", [])
        
        if "X" not in platforms:
            return {}
        
        print("\n[X] Posting to X (Twitter)...")
        
        if not self.x_client or not self.x_api_v1:
            print("   [ERROR] X credentials not configured")
            return {"posting_urls": {"X": "Error: Credentials not configured"}}
        
        approved_ads = state.get("approved_ads", [])
        x_ad = _ad_find(approved_ads, "X")

        if not x_ad:
            print("   [ERROR] No approved X ad found")
            return {"posting_urls": {"X": "Error: No approved ad found"}}

        message = f"{_ad_get(x_ad, 'headline')}\n\n{_ad_get(x_ad, 'text')}\n\n"
        message += " ".join((_ad_get(x_ad, 'hashtags') or [])[:3])
        message += f"\n\nApply: {state.get('job_form_link', '')}"
        
        if len(message) > 280:
            message = message[:277] + "..."
        
        try:
            temp_image_path = state.get("temp_image_path")

            # If it's an S3/HTTP URL, download to a local temp file for X media upload
            local_temp = None
            if temp_image_path and (temp_image_path.startswith("http://") or temp_image_path.startswith("https://")):
                local_temp = self._download_image_for_x(temp_image_path)
            elif temp_image_path and os.path.exists(temp_image_path):
                local_temp = temp_image_path

            if local_temp:
                print("   [INFO] Uploading image to X...")
                media = self.x_api_v1.media_upload(local_temp)
                if temp_image_path and temp_image_path.startswith("http"):
                    self._cleanup_temp_image(local_temp)
                status = self.x_api_v1.update_status(status=message, media_ids=[media.media_id])
                post_url = f"https://x.com/user/status/{status.id}"
                print(f"   [SUCCESS] Tweet ID: {status.id}")
                print(f"   [SUCCESS] URL: {post_url}")
                return {"posting_urls": {"X": post_url}}
            else:
                print("   [INFO] Posting text-only...")
                response = self.x_client.create_tweet(text=message)
                post_url = f"https://x.com/user/status/{response.data['id']}"
                print(f"   [SUCCESS] Tweet ID: {response.data['id']}")
                print(f"   [SUCCESS] URL: {post_url}")
                return {"posting_urls": {"X": post_url}}
                
        except Exception as e:
            print(f"   [ERROR] Exception: {e}")
            return {"posting_urls": {"X": f"Error: {str(e)}"}}

    def collect_results_node(self, state: JDCreationState):
        """Node 5: Collect all posting results"""
        posting_urls = state.get("posting_urls", {})
        print(f"\n[COLLECT] Collected posting URLs from state: {posting_urls}")
        print(f"[COLLECT] Number of platforms with results: {len(posting_urls)}")
        for platform, url in posting_urls.items():
            print(f"[COLLECT]   - {platform}: {url[:80]}{'...' if len(url) > 80 else ''}")
        
        # Ensure the collected URLs are returned to maintain state
        return {}
    
    def cleanup_node(self, state: JDCreationState):
        """Node 6: Cleanup temporary files"""
        temp_path = state.get("temp_image_path")
        if temp_path:
            self._cleanup_temp_image(temp_path)
        return {}

    def summary_node(self, state: JDCreationState):
        """Node 7: Generate summary and final status"""
        print("\n" + "="*60)
        print("AD POSTING SUMMARY")
        print("="*60)
        
        platforms = state.get("platforms", [])
        posting_urls = state.get("posting_urls", {})
        
        print(f"[SUMMARY] posting_urls in final state: {posting_urls}")
        print(f"[SUMMARY] posting_urls type: {type(posting_urls)}")
        print(f"[SUMMARY] posting_urls length: {len(posting_urls)}")
        
        print(f"Platforms targeted: {', '.join(platforms)}")
        print(f"Approved ads: {len(state.get('approved_ads', []))}")
        print(f"Job form link: {state.get('job_form_link', 'N/A')}")
        
        if posting_urls:
            print("\nPosted URLs:")
            for platform, url in posting_urls.items():
                print(f"  - {platform}: {url}")
        else:
            print("\n[WARNING] No posting_urls found in state!")
        
        print("="*60)
        
        # Ensure posting_urls is explicitly returned in the final state
        return {
            "orchestration_status": "AD_Posting_Agent",
            "status": "done"
        }

