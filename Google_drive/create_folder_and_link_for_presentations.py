import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------------------- Step 1: Authenticate with Google Drive ---------------------------- #
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'talkppt-af5c2fc084ba.json'  # Path to your JSON file

# Authenticate and build the service
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# ---------------------------- Step 2: Folder Creation ---------------------------- #
def create_folder(folder_name, parent_folder_id=None):
    """
    Create a folder in Google Drive under a specified parent folder.
    Returns the folder ID if successful.
    """
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
    }
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]
    
    # Create folder
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')

def set_folder_permission(folder_id):
    """
    Set the folder permissions so anyone with the link can view.
    """
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    drive_service.permissions().create(
        fileId=folder_id,
        body=permission,
        fields='id'
    ).execute()

# ---------------------------- Step 3: Main Script Logic ---------------------------- #
def main():
    # Parent folder ID where all new folders will be created
    parent_folder_id = '1ZB81tscq38qqMB3WKKlq4ND_jMltAI9C'  # Replace with your parent folder ID
    
    # Read the input file (tab-separated with no headers)
    df = pd.read_csv('Names.txt', sep='\t', header=None, names=['First Name', 'Last Name', 'Day', 'Room'])
    
    # ------------------ Step 1: Create Day Folders ------------------ #
    day_folders = df['Day'].drop_duplicates()
    day_folder_ids = {}
    
    for day in day_folders:
        day_folder_id = create_folder(day, parent_folder_id)
        set_folder_permission(day_folder_id)  # Share the day folder
        day_folder_ids[day] = day_folder_id
        print(f"Created folder for {day}")

    # ------------------ Step 2: Create Room Folders inside each Day folder ------------------ #
    room_folders = df[['Day', 'Room']].drop_duplicates()
    room_folder_ids = {}

    for _, row in room_folders.iterrows():
        day = row['Day']
        room = row['Room']
        
        # Get the parent day folder ID
        day_folder_id = day_folder_ids[day]
        
        # Create room folder inside the day folder
        room_folder_id = create_folder(room, day_folder_id)
        set_folder_permission(room_folder_id)  # Share the room folder
        room_folder_ids[(day, room)] = room_folder_id
        print(f"Created room folder {room} inside {day}")

    # ------------------ Step 3: Assign shared link to each person ------------------ #
    results = []
    
    for _, row in df.iterrows():
        day = row['Day']
        room = row['Room']
        
        # Get the room folder ID based on Day and Room combination
        room_folder_id = room_folder_ids[(day, room)]
        
        # Generate the shared link for the folder
        shared_link = f"https://drive.google.com/drive/folders/{room_folder_id}"
        
        # Add the link for this person
        results.append({
            'First Name': row['First Name'],
            'Last Name': row['Last Name'],
            'Day': day,
            'Room': room,
            'Shared Link': shared_link
        })
        print(f"Assigned shared link for {row['First Name']} {row['Last Name']} -> Link: {shared_link}")

    # ------------------ Step 4: Save results to a CSV file ------------------ #
    output_df = pd.DataFrame(results)
    output_df.to_csv('updated_names_with_links.csv', index=False)
    print("Folders created, and shared links saved to 'updated_names_with_links.csv'.")

# ---------------------------- Run the Script ---------------------------- #
if __name__ == '__main__':
    main()
