package com.reversi.client;

import com.reversi.common.EventBus;
import com.reversi.common.JacksonObjMapper;
import com.reversi.common.Message;
import java.io.*;
import java.net.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ServerSocket implements IController {
  private static final Logger logger =
      LoggerFactory.getLogger(ServerSocket.class);

  private EventBus eventBus;
  private Socket socket;
  private PrintWriter out;
  private BufferedReader in;

  public ServerSocket(EventBus eventBus) { this.eventBus = eventBus; }

  // Connect to the server (assumed to be running on localhost:5000)
  public void connectToServer() {
    try {
      socket = new Socket("localhost", 5000);
      out = new PrintWriter(socket.getOutputStream(), true);
      in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
      // Listen for messages from the server on a separate thread.
      new Thread(() -> listenToServer()).start();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  // Continuously listen for server messages and update the view accordingly.
  private void listenToServer() {
    try {
      String line;
      while ((line = in.readLine()) != null) {
        logger.info("Received: {}", line);
        Message msg = JacksonObjMapper.get().readValue(line, Message.class);
        eventBus.post(new ServerMessage(msg));
      }
    } catch (IOException e) {
      logger.error("Error listening to server", e);
    } catch (Exception e) {
      logger.error("Unknown error occured while processing data from server");
    }
  }

  public void send(Message msg) {
    if (out == null) {
      logger.error("Output stream is null, message discarded: {}", msg);
      return;
    }

    try {
      out.println(JacksonObjMapper.get().writeValueAsString(msg));
    } catch (Exception e) {
      logger.error("Failed to send: {}", msg);
    }
  }

  // Called by the view when a cell is clicked.
  public void sendMove(int row, int col) {
    send(new Message(new Message.Move(row, col)));
  }
}
