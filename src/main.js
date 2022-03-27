import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './index.css'

import { initializeApp } from 'firebase/app';
import { getFirestore } from "firebase/firestore";
import { getAuth, browserSessionPersistence } from "firebase/auth";

// this config is ripped from on-call. need to change to cardiacml

const firebaseConfig = {
  apiKey: "AIzaSyCuHnGCmlKYQGBFFuKRsbz0xLQHomk-tyQ",
  authDomain: "on-call-5e00c.firebaseapp.com",
  projectId: "on-call-5e00c",
  storageBucket: "on-call-5e00c.appspot.com",
  messagingSenderId: "306712776551",
  appId: "1:306712776551:web:3e4b4e9dc5aadeb25da9c2",
  measurementId: "G-LHTC415CXT"
};

const firestoreApp = initializeApp(firebaseConfig);

export const db = getFirestore(firestoreApp);
const auth = getAuth(firestoreApp);
auth.setPersistence(browserSessionPersistence);
export { auth }

createApp(App)
  .use(router)
  .mount('#app')
