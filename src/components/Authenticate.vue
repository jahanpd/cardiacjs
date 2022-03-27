<template>
  <form class="p-3 gap-4">
    <div v-if="!authenticated" class="flex flex-row justify-center">
        <ul class="flex flex-wrap -mb-px">
          <li class="mr-2" v-on:click="mode='SignIn'">
                <a v-on:click="page=1" class="inline-block py-4 px-4 text-sm font-medium text-center text-gray-500 rounded-t-lg border-b-2 border-transparent hover:text-gray-600 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300">
                  Sign In
                </a>
          </li>
          
          <li class="mr-2" v-on:click="mode='SignUp'">
                <a v-on:click="page=2" class="inline-block py-4 px-4 text-sm font-medium text-center text-gray-500 rounded-t-lg border-b-2 border-transparent hover:text-gray-600 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300">
                  Sign Up
                </a>
          </li>
          <li class="mr-2" v-on:click="mode='Reset'">
                <a v-on:click="page=2" class="inline-block py-4 px-4 text-sm font-medium text-center text-gray-500 rounded-t-lg border-b-2 border-transparent hover:text-gray-600 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300">
                  Forgot Password
                </a>
          </li>


        </ul>
    </div>

    <div v-if="!authenticated" class="flex flex-row p-3 justify-center">
      <p class="w-32" >Email: </p>
      <input
          v-model="user.email"
          class="flex shadow appearance-none w-80 border rounded px-2 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" 
          type="email" 
          placeholder="email"
          required>
    </div>

    <div v-if="!authenticated & mode !== 'Reset'" class="flex flex-row p-3 justify-center">
      <p class="w-32" >Password: </p>
      <input
          v-model="user.password"
          class="flex shadow appearance-none w-80 border rounded px-2 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" 
          type="password" 
          placeholder="password"
          required>
    </div>

    <div v-if="!authenticated & mode === 'SignUp'" class="flex flex-row px-5 justify-center">
      <p class="w-32" >Password: </p>
      <input
          v-model="user.repeat"
          class="flex shadow appearance-none w-80 border rounded px-2 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" 
          type="password" 
          placeholder="repeat password"
          required>
    </div>
    
    <div v-if="!authenticated & mode==='SignIn'" class="flex flex-row p-5 justify-center">
      <input 
        type="submit"
        class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        v-on:click="login(user.email, user.password)"
        value="login"
      >
    </div>
    <div v-if="!authenticated & mode==='SignUp'" class="flex flex-row p-5 justify-center">
      <input
        type="submit"
        value="sign up"
        v-on:click="signup(user.email, user.password, user.repeat)"
        class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
    </div>

    <div v-if="!authenticated & mode==='Reset'" class="flex flex-row p-5 justify-center">
      <input
        type="submit"
        value="submit"
        v-on:click="resetpwd(user.email)"
        class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
    </div>


    <div v-if="authenticated" class="flex flex-row p-5 justify-center">
      <button 
        class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        v-on:click="logout()"
        >
        logout
      </button>
    </div>
  </form>
</template>

<script>
import { auth } from '../main'
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, sendPasswordResetEmail, signOut} from "firebase/auth";

export default {
  name: 'Authenticate',
  data(){
    return {
      user: {
        name:"",
        email:"",
        password:"",
        repeat:"",
      },
      errorMsg:"",
      mode: "SignIn",
      authenticated: false,
    }
  },
  methods: {
    login(email, password){
      try {
        if (!email || !password) {
          this.errorMsg = "Email and Password Required";
          return;
        }
        signInWithEmailAndPassword(
            auth,
            email,
            password)
          .then((userCredential) => {
              this.authenticated = true;
              this.$router.push("/");
          })
          .catch((error) => {
            this.errorMsg = error.message;
          });
      } catch (error) {
        this.errorMsg = "Login Error"
      };
    },
    signup(email, password, repeat){
      alert("Not allowing new signup at this stage")
    },
    resetpwd(email){
      console.log(email)
      sendPasswordResetEmail(auth, email).then(() => {
        alert("Password reset email sent")
      }).catch((error) => {
          console.log(error);
          this.errorMsg = error.message;
      });
    },
    logout(){
      signOut(auth).then(() => {
        console.log(this.authenticated);
        this.authenticated = false;
      }).catch((error) => {
        console.log(error);
        this.authenticated = false;
      });
    },
  },
  beforeCreate(){
    try {
      auth.onAuthStateChanged((user) => {
        if (user) {
          this.authenticated = true;
        }
        else {
          console.log('no user registered');
          this.authenticated = false;
        }
      })
    } catch (error) {
      this.authenticated = false;
    }
  },
}
</script>
