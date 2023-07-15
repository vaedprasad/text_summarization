# Rad Text Summarization Service

## Instructions
1. Clone the repository locally via either HTTPS or SSH:
	- HTTPS: `git clone https://github.com/vaedprasad/text_summarization.git`
	- SSH: `git clone git@github.com:vaedprasad/text_summarization.git`
	
2. Enter the root directory of the repository
	- `cd ~/text_summarization`
	
3. Build images and start containers
	- `docker compose up --build` (If you do not have Docker installed, install [here](https://docs.docker.com/get-docker/)
	- Wait for building to complete (Est. 20 - 30 seconds)
 	- Wait for application to launch (You should see `Application startup complete.` in your terminal)
	
4. Open FastAPI Application on Browser
	- Navigate to `http://127.0.0.1:8000/docs` on a preferred browser
	
5. Read API Description and have fun with the service!
	- *Note*: After exiting the service, you can restart the service with `docker compose up`
