# !pip install pushbullet.py
from IPython.core.magic import Magics, magics_class, line_magic
from pushbullet import Pushbullet
import json

def load_config():
    with open('data/token_pushbullet.json', 'r') as f:
        config = json.load(f)
    return config

@magics_class
class PushbulletMagic(Magics):
    def __init__(self, shell):
        super(PushbulletMagic, self).__init__(shell)
        self.pb = Pushbullet( load_config()["pushbullet_api_key"] ) 
    
    @line_magic
    def pushbullet_notify(self, line):
        """
        %pushbullet_notify MESSAGE
        Send a Pushbullet notification with the provided message.
        """
        message = line.strip()
        title = 'Your Jupyter Message'
        
        push = self.pb.push_note(title, message)
        print("Message sent.")

# Register the magic
get_ipython().register_magics(PushbulletMagic)