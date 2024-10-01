# Chainlit Assistant

This is a template for quick setup of an OpenAI Assistant using Chainlit.

It is based on [Chainlit\'s example repo](https://github.com/Chainlit/openai-assistant/) as well as [this fork](https://github.com/renyuantime/openai-assistant/tree/feature/add-tool-call) that adds the capacity for custom tool calls.

### How to launch it

- Set up an OpenAI API key and create an Assistant (e.g. in the playground).
- `cp .env.example .env`
- Open `.env` and add the keys and IDs you need. _(Note: You may not need the project or organization IDs if you're using the defaults.)_
- `pip install -r requirements.txt`
- `chainlit run app.py -w`

Now it should be up and running.

Support for attachments, and image and audio generation, are available but not heavily tested. Please try them, make improvements, give feedback. etc.

### Adding tools

Add any custom functions to `tools.py` and define them in the `CUSTOM_TOOLS` object following the example.

You will also need to define the function in the Assistant playground.