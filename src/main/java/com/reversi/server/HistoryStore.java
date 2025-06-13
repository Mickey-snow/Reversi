package com.reversi.server;

import com.fasterxml.jackson.core.type.TypeReference;
import com.reversi.common.GameRecord;
import com.reversi.common.JacksonObjMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Simple file-based storage for completed game records. */
public class HistoryStore {
  private static final Logger logger = LoggerFactory.getLogger(HistoryStore.class);
  private static final String FILE_NAME = "game_history.json";

  private final List<GameRecord> history = new ArrayList<>();

  public HistoryStore() { load(); }

  private void load() {
    File f = new File(FILE_NAME);
    if (!f.exists()) return;
    try {
      List<GameRecord> data =
          JacksonObjMapper.get()
              .readValue(f, new TypeReference<List<GameRecord>>() {});
      history.addAll(data);
    } catch (IOException e) {
      logger.error("Failed to load history", e);
    }
  }

  private void save() {
    File f = new File(FILE_NAME);
    try {
      JacksonObjMapper.get().writerWithDefaultPrettyPrinter().writeValue(f, history);
    } catch (IOException e) {
      logger.error("Failed to save history", e);
    }
  }

  public synchronized void addRecord(GameRecord rec) {
    history.add(rec);
    save();
  }

  public synchronized List<GameRecord> getHistory() {
    return new ArrayList<>(history);
  }
}
