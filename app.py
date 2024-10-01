import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from typing_extensions import override
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI, AsyncAssistantEventHandler
from openai.types.beta.threads.runs import RunStep

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element

from tools import CUSTOM_TOOLS

load_dotenv()


# Async OpenAI client
sync_openai_client = OpenAI()
async_openai_client = AsyncOpenAI()

assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)
config.ui.name = assistant.name


# Define event handler for Chainlit interaction
class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.assistant_name = assistant_name
        self.current_message = None
        self.current_step = None
        self.current_tool_call = None
        self.function_map = CUSTOM_TOOLS

    @override
    async def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action',since these will have our tool_calls
        if event.event == 'thread.run.requires_action':
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.current_run.id=run_id
            await self.handle_requires_action(event.data, run_id)

    async def handle_requires_action(self, data, run_id):
        tool_outputs = []
        for tool in data.required_action.submit_tool_outputs.tool_calls:
            func_name = tool.function.name
            func_args = tool.function.arguments

            # Use the instance attribute function_map
            func_to_call = self.function_map.get(func_name)
            
            if func_to_call:
                try:
                    # Parse the func_args JSON string to a dictionary
                    func_args_dict = json.loads(func_args)
                    tool_call_output = func_to_call(**func_args_dict)
                    tool_outputs.append({"tool_call_id": tool.id, "output": tool_call_output})
                except TypeError as e:
                    print(f"Error calling function {func_name}: {e}")
            else:
                print(f"Function {func_name} not found")
        # Submit all tool_outputs at the same time
        await self.submit_tool_outputs(tool_outputs, run_id)

    async def submit_tool_outputs(self, tool_outputs, run_id):
        """
        Submits the tool outputs to the current run.
        """
        async with async_openai_client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(assistant_name=self.assistant_name),
        ) as stream:
            await stream.until_done()
    
    async def on_text_created(self, text):
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        await self.current_message.update()

    async def on_tool_call_delta(self, delta, snapshot): 
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = datetime.now(timezone.utc).isoformat() + 'Z'
            #await self.current_step.send()  

        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(
                            name=delta.type,
                            type="tool"
                        )
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = datetime.now(timezone.utc).isoformat() + 'Z'
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)

    async def on_run_step_done(self, run_step: RunStep):
        if run_step.type == 'tool_calls':
            tool_calls = run_step.step_details.tool_calls

            # Handle tool call with type 'file_search and output quations to user'
            if any(call.type == 'file_search' for call in tool_calls):
                #retrieve quations from openai by adding parameter include
                run_step = sync_openai_client.beta.threads.runs.steps.retrieve(
                    thread_id=cl.user_session.get("thread_id"),
                    run_id=run_step.run_id,
                    step_id=run_step.id,
                    include=["step_details.tool_calls[*].file_search.results[*].content"]
                    )
                # Initialize an empty list to hold the citations
                citations = []
                # Extract tool_calls from run_step.step_details
                tool_calls = run_step.step_details.tool_calls
                # Iterate through each tool call
                for call in tool_calls:
                    # Check if the type of the tool call is 'file_search'
                    if call.type == 'file_search':
                        # Extract the file search results from the file_search attribute
                        file_search_results = call.file_search.results
                        # Iterate through each result in file_search_results
                        for result in file_search_results:
                            # Extract the first content's text for the quote (if available)
                            quote = result.content[0].text if result.content else ""
                            # Create a citation dictionary
                            citation = {
                                "file_name": result.file_name,
                                "score": result.score,
                                "quote": quote
                            }
                            # Append the citation dictionary to the citations list
                            citations.append(citation)

                # Create the final dictionary with the list of citations
                citations_dict = {"citations": citations}
                #Construct step 
                self.current_step = cl.Step(name=call.type, type="tool")
                self.current_step.output=citations_dict
                await self.current_step.send()

            #Handle tool calls that are customized functions and render function input and output to user
            else:
                for tool_call in tool_calls:
                    if hasattr(tool_call, 'function'):
                        func_name = tool_call.function.name
                        func_args = tool_call.function.arguments
                        func_output=tool_call.function.output     
                        self.current_step = cl.Step(name=func_name, type="tool")
                        self.current_step.input = json.loads(func_args)

                        # Handle func_output being either JSON string or plain text
                        try:
                            # Try to parse func_output as JSON
                            self.current_step.output = json.loads(func_output)
                        except json.JSONDecodeError:
                            # If it's not JSON, treat it as plain text
                            self.current_step.output = func_output
                        await self.current_step.send()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id,
            content=response.content,
            display="inline",
            size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids



async def process_files(files: List[Element]):
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}],
        }
        for file_id in file_ids
    ]


# Async chat start handler
@cl.on_chat_start
async def start_chat():
    thread = await async_openai_client.beta.threads.create()
    cl.user_session.set('thread_id', thread.id)

    await cl.Message(content=f"Hello, I'm {assistant.name}!").send()


# Main message handler
@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get('thread_id')

    attachments = await process_files(message.elements)

    # Create and send a user message
    oai_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments
    )

    # Create and stream a run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name)
    ) as stream:
        await stream.until_done()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[Element]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg)
