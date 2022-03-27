<template>
  <div class="flex flex-row justify-center p-5 flex-wrap md:flex-nowrap">

    <div class="flex p-5 md:w-1/2 md:justify-end">
      <p class="md:text-right md:w-1/2">
        Deep learning for risk prediction in cardiac surgery.
        Made possible with the guidance and support of the amazing ANZSCTS National Audit Database team and Monash University
      </p>
    </div>

    <div class="flex md:w-1/2 justify-center md:justify-start p-5 md:w-1/2">
      <img src="../assets/smaller.png" class="md:w-1/2"/>
    </div>

  </div>

  <div class="flex flex-grow"/> 
  <div class="flex flex-row justify-center p-5 align-text-bottom">
    <p class="text-xs w-full md:w-3/12 align-text-bottom p-5 text-right md:text-right">
        Developed as part of the doctoral research of Dr JC Penny-Dimri under the supervision Dr C Bergmeir and Prof J Smith
      </p>

      <div class="flex md:w-1/2 justify-start p-5 md:w-3/12">
        <img src="../assets/monash.png" class="max-h-[8rem] max-w-[8rem]"/>
      </div>
  </div>
</template>

<script>
export default {
  name: 'Landing',
  data: () => ({

  })
}
</script>
