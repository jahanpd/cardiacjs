import Dexie from 'dexie';

export const db = new Dexie('models');
db.version(1).stores({
  models: '++id, datetime, name, config, weights', // Primary key and indexed props
});
