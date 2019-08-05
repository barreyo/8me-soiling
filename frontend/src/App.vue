<template>
  <v-app id="inspire">
    <v-content>
      <v-container fluid fill-height>
        <v-layout align-center justify-center>
          <v-flex xs12 sm12 md6>
            <v-card class="elevation-6">
              <v-toolbar color="primary" dark flat>
                <v-toolbar-title>Solar Soiling Calculator</v-toolbar-title>
                <v-spacer></v-spacer>
              </v-toolbar>
              <v-card-text class="pa-10">
                <v-form>
                  <v-textarea solo name="input-7-4" label="Precipitation Data"></v-textarea>
                  <v-text-field value="0.1" label="Soiling Rate" suffix="%"></v-text-field>
                  <v-text-field value="5.0" label="Precipitation Threshold"></v-text-field>
                  <v-text-field value="1.0" label="Precipitation Wash Floor"></v-text-field>
                  <v-text-field value="0.0" label="Manual Wash Floor"></v-text-field>
                  <v-text-field value="10" type="number" label="Manual Wash Grace Period"></v-text-field>
                  <v-layout>
                    <v-flex class="pr-4">
                      <v-subheader class="pl-0">Average Washes Per Year</v-subheader>
                      <v-slider
                        v-model="slider"
                        class="align-center"
                        max="100"
                        min="0"
                        value="10"
                        thumb-label
                      >
                        <template v-slot:append>
                          <v-text-field
                            v-model="slider"
                            class="mt-0 pt-0"
                            single-line
                            value="10"
                            type="number"
                            style="width: 60px"
                          ></v-text-field>
                        </template>
                      </v-slider>
                    </v-flex>
                  </v-layout>
                </v-form>
              </v-card-text>
              <v-card-actions class="pl-10 pr-10 pb-8">
                <v-spacer></v-spacer>
                <v-btn color="primary">Download</v-btn>
              </v-card-actions>
            </v-card>
          </v-flex>
        </v-layout>
      </v-container>
    </v-content>
  </v-app>
</template>

<script>
export default {
  name: "App",
  components: {},
  data: () => ({
    slider: 10
  })
};
</script>
