package com.reversi.server;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import com.reversi.server.HistoryStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ServerMain {
  private static final Logger logger =
      LoggerFactory.getLogger(ServerMain.class);
  public static final int PORT = 5000;

  public static void main(String[] args) { new ServerMain().startServer(); }

  // ExecutorService for managing client tasks
  private final ExecutorService clientThreadPool =
      Executors.newCachedThreadPool();
  // SessionHub takes care of all server-client messaging and game state updates.
  private final HistoryStore historyStore = new HistoryStore();
  private final SessionHub session = new SessionHub(historyStore);

  private static final AtomicInteger clientCounter = new AtomicInteger(0);
  private int genClientId() { return clientCounter.incrementAndGet(); }

  public void startServer() {
    try (ServerSocket serverSocket = new ServerSocket(PORT)) {
      logger.info("Server started on port {}", PORT);
      while (true) {
        Socket socket = serverSocket.accept();
        int clientId = genClientId();
        ClientSocket handler =
            new ClientSocket(clientId, socket, session.getEventBus());
        session.registerClient(handler);
        clientThreadPool.submit(handler);
        logger.info("Client connected. Assigned client ID: {}", clientId);
      }
    } catch (IOException e) {
      logger.error("Error starting server", e);
    } finally {
      clientThreadPool.shutdown();
    }
  }
}
