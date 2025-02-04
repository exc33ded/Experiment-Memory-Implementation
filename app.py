from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from models import db, User, Project, LongTermMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import os
import markdown

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# LangChain configuration
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
chat_model = ChatGoogleGenerativeAI(model="gemini-pro")

# In-memory store (for temporary memory storage)
memory_store = {}

# Convert markdown content to HTML
def render_markdown(content):
    return markdown.markdown(content)

@app.route('/')
def home():
    projects = Project.query.all()
    return render_template('index.html', projects=projects)

# Create a project and display the list of projects
@app.route('/create_project', methods=['POST'])
def create_project():
    data = request.form
    project = Project(
        id=data['project_id'],
        name=data['name'],
        summary=data['summary'],
        user_id=data['user_id']
    )
    db.session.add(project)
    db.session.commit()
    return redirect(url_for('view_project', project_id=project.id))

# Buffer memory configuration (keeps the last N messages)
BUFFER_SIZE = 40  # Set buffer size to retain the last 5 messages

def get_memory(project_id):
    memory = memory_store.get(project_id)
    if not memory:
        # Initialize in-memory storage for this project
        memory = {'messages': []}
        memory_store[project_id] = memory

        # Retrieve memory from the database
        long_memory = LongTermMemory.query.filter_by(project_id=project_id).first()
        if long_memory:
            # Load chat history from the database into memory
            for line in long_memory.chat_content.split('\n'):
                if '|' in line:
                    try:
                        sender, content = line.split('|', 1)
                        memory['messages'].append({'sender': sender, 'content': content})
                    except ValueError:
                        print(f"Skipping malformed line: {line}")
        else:
            # No previous memory, create a new record
            new_memory = LongTermMemory(
                project_id=project_id,
                user_id=Project.query.get(project_id).user_id,
                chat_content=""
            )
            db.session.add(new_memory)
            db.session.commit()
    return memory

# Add a message to the buffer (ensures we only keep the last N messages)
def add_message_to_buffer(memory, sender, content):
    memory['messages'].append({'sender': sender, 'content': content})
    if len(memory['messages']) > BUFFER_SIZE:
        memory['messages'].pop(0)  # Remove the oldest message to maintain buffer size

@app.route('/project/<int:project_id>', methods=['GET', 'POST'])
def view_project(project_id):
    project = Project.query.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found!'}), 404

    memory = get_memory(project_id)
    chat_history = memory['messages']

    if request.method == 'POST':
        user_message = request.form.get('message')

        if user_message and user_message.strip():
            # Add user message to memory buffer
            add_message_to_buffer(memory, 'user', render_markdown(user_message))

            # Retrieve past messages (history) for context
            previous_conversations = "\n".join(
                [f"{msg['sender'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]]
            )

            # Create the prompt including past conversations
            prompt = f"""
            You are an AI assistant helping users with project-related queries. The project details are as follows:

            Project Summary:
            {project.summary}

            Previous Conversations:
            {previous_conversations}

            The user has asked the following question: "{user_message}"
            """

            # Get AI response based on the history and current user message
            ai_response = chat_model.invoke([HumanMessage(content=prompt)]).content or \
                          f"Here’s a bit more information about the project: {project.summary}."

            # Add AI response to memory buffer
            add_message_to_buffer(memory, 'ai', render_markdown(ai_response))

            # Save updated memory to the database
            chat_history_str = "\n".join([f"{msg['sender']}|{msg['content']}" for msg in memory['messages']])
            long_memory = LongTermMemory.query.filter_by(project_id=project_id).first()
            if long_memory:
                long_memory.chat_content = chat_history_str
            else:
                long_memory = LongTermMemory(
                    project_id=project_id,
                    user_id=project.user_id,
                    chat_content=chat_history_str
                )
                db.session.add(long_memory)
            db.session.commit()

        return jsonify({'chat_history': chat_history})

    # Render the chat page for GET request
    return render_template('chat.html', project=project, summary=project.summary, chat_history=chat_history)

@app.route('/save_chat/<project_id>', methods=['POST'])
def save_chat(project_id):
    project = Project.query.get(project_id)
    if not project:
        return render_template('error.html', message="Project not found!")

    memory = get_memory(project_id)
    chat_history = memory['messages']

    # Save the full chat history to the database
    if chat_history:
        # Ensure all messages are in valid format
        chat_history_str = "\n".join([f"{msg['sender']}|{msg['content']}" 
            for msg in chat_history if 'sender' in msg and 'content' in msg
        ])

        # Save to LongTermMemory
        long_memory = LongTermMemory.query.filter_by(project_id=project_id).first()
        if long_memory:
            long_memory.chat_content = chat_history_str
        else:
            long_memory = LongTermMemory(
                project_id=project_id,
                user_id=project.user_id,
                chat_content=chat_history_str
            )
            db.session.add(long_memory)
        db.session.commit()

        # Clear in-memory messages after saving
        memory_store[project_id]['messages'] = []

    return redirect(url_for('view_project', project_id=project.id))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
